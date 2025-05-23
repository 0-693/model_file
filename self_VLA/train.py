import dataclasses
import logging
import os
import re
from typing import Protocol, runtime_checkable, List, Union

import flax.nnx as nnx
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import io
import optax # For optimizers

from models import Pi0, Pi0Config, Observation # Assuming Pi0 and Pi0Config are in models.py
from shared import array_typing as at
import shared.download as download
from models import PaligemmaTokenizer # Import PaligemmaTokenizer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Utility functions from paligemma_pi0_example.py ---
@runtime_checkable
class WeightLoader(Protocol):
    def load(self, params: at.Params) -> at.Params:
        """Loads model weights."""
        ...

@dataclasses.dataclass(frozen=True)
class PaliGemmaWeightLoader(WeightLoader):
    """Loads weights from official PaliGemma checkpoints."""
    params_path: str

    def load(self, params: at.Params) -> at.Params:
        path = download.maybe_download(
            self.params_path, gs={"token": "anon"}
        )
        with path.open("rb") as f:
            flat_params_np = dict(np.load(f, allow_pickle=False))

        unflattened_all_params = flax.traverse_util.unflatten_dict(flat_params_np, sep="/")
        
        if "params" not in unflattened_all_params:
            raise ValueError(
                f"Expected 'params' key in loaded checkpoint, but found: {unflattened_all_params.keys()}"
            )

        loaded_params_for_merge = {"PaliGemma": unflattened_all_params["params"]}
        
        return _merge_params(loaded_params_for_merge, params, missing_regex=".*")

def _merge_params(loaded_params: at.Params, params: at.Params, *, missing_regex: str) -> at.Params:
    """Merges loaded parameters with reference parameters."""
    flat_ref = flax.traverse_util.flatten_dict(params, sep="/")
    flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")

    result = {}
    for k, v in flat_loaded.items():
        if k in flat_ref:
            result[k] = jnp.asarray(v, dtype=flat_ref[k].dtype)
        else:
            logger.warning(
                f"Loaded parameter key '{k}' not found in reference model parameters. Skipping."
            )

    pattern = re.compile(missing_regex)
    for k in {k for k in flat_ref if pattern.fullmatch(k)}:
        if k not in result:
            result[k] = jnp.asarray(flat_ref[k])

    return flax.traverse_util.unflatten_dict(result, sep="/")

def load_image(image_bytes, target_size=(224, 224)):
    """Loads and preprocesses an image from bytes."""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize(target_size, Image.LANCZOS)
    image_array = np.array(image).astype(np.float32) / 127.5 - 1.0
    return image_array

# --- Dataset Class (Adapted from rlds_dataset.py) ---
class RLDSParquetDataset:
    def __init__(
        self,
        parquet_paths: Union[str, List[str]],
        image_size: int = 224,
        max_samples_per_file: Union[int, None] = None,
        use_both_views: bool = True,
        action_horizon: int = 16,
        tokenizer: PaligemmaTokenizer = None, # Pass tokenizer here
        max_token_len: int = 48, # Needed for tokenizer padding
    ):
        self.data = []
        self.image_size = image_size
        self.use_both_views = use_both_views
        self.action_horizon = action_horizon
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len

        if isinstance(parquet_paths, str) and os.path.isdir(parquet_paths):
            file_list = sorted(glob.glob(os.path.join(parquet_paths, "*.parquet")))
        elif isinstance(parquet_paths, list):
            file_list = parquet_paths
        else:
            raise ValueError("parquet_paths should be a directory path or a list of files.")

        for path in tqdm(file_list, desc="? Loading Parquet"):
            df = pd.read_parquet(path)
            if "frame" in df.columns:
                df = df.sort_values("frame").reset_index(drop=True)

            if max_samples_per_file is not None:
                df = df.head(max_samples_per_file)

            for i in range(len(df) - action_horizon + 1):
                sub_df = df.iloc[i : i + action_horizon]
                episode = {
                    "image_1": sub_df.iloc[0]["image_1"],
                    "image_2": sub_df.iloc[0]["image_2"],
                    "prompt": sub_df.iloc[0]["prompt"],
                    "state": sub_df.iloc[0]["state"],
                    "action": [row["action"] for _, row in sub_df.iterrows()],
                }
                self.data.append(episode)
        
        print(f"Total parquet files: {len(file_list)}")
        print(f"Loaded dataset with {len(self.data)} samples.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        images = {}
        images["base_0_rgb"] = load_image(item["image_1"], target_size=(self.image_size, self.image_size))
        image_masks = {"base_0_rgb": True}

        if self.use_both_views:
            images["wrist_0_rgb"] = load_image(item["image_2"], target_size=(self.image_size, self.image_size))
            image_masks["wrist_0_rgb"] = True

        prompt = item["prompt"]
        if isinstance(prompt, bytes):
            prompt = prompt.decode("utf-8")

        # Tokenize prompt using PaligemmaTokenizer
        tokens, mask = self.tokenizer.tokenize(prompt)

        state = jnp.asarray(item["state"], dtype=jnp.float32)
        action = jnp.asarray(np.stack(item["action"]), dtype=jnp.float32)

        return {
            "images": images,
            "image_masks": image_masks,
            "prompt": prompt, # Keep prompt for debugging, but tokens are used for model input
            "tokenized_prompt": tokens,
            "tokenized_prompt_mask": mask,
            "state": state,
            "action": action,
        }

def custom_collate_fn(batch_list):
    """Custom collate function to handle variable length lists of images and prompts."""
    # Stack states, actions, and tokenized data directly
    states = jnp.stack([item["state"] for item in batch_list], axis=0)
    actions = jnp.stack([item["action"] for item in batch_list], axis=0)
    tokenized_prompts = jnp.stack([item["tokenized_prompt"] for item in batch_list], axis=0)
    tokenized_prompt_masks = jnp.stack([item["tokenized_prompt_mask"] for item in batch_list], axis=0)

    # Handle images - assuming consistent camera views across batch
    first_item_images = batch_list[0]["images"]
    batched_images = {
        cam_name: jnp.stack([item["images"][cam_name] for item in batch_list], axis=0)
        for cam_name in first_item_images
    }
    
    first_item_image_masks = batch_list[0]["image_masks"]
    batched_image_masks = {
        cam_name: jnp.stack([item["image_masks"][cam_name] for item in batch_list], axis=0)
        for cam_name in first_item_image_masks
    }

    return {
        "images": batched_images,
        "image_masks": batched_image_masks,
        "tokenized_prompt": tokenized_prompts,
        "tokenized_prompt_mask": tokenized_prompt_masks,
        "state": states,
        "action": actions,
    }


# --- Training Function ---
def train(config):
    # Initialize JAX RNG
    key = jax.random.key(0)

    # Create Pi0 model configuration
    pi0_config = Pi0Config(
        action_dim=config["action_dim"],
        action_horizon=config["action_horizon"],
        max_token_len=config["max_token_len"],
        paligemma_variant=config["paligemma_variant"],
        action_expert_variant=config["action_expert_variant"],
        dtype=config["dtype"]
    )

    logger.info("Creating Pi0 model...")
    model_key, dropout_key = jax.random.split(key)
    model = pi0_config.create(model_key)

    # Load pre-trained PaliGemma weights
    if config.get("load_paligemma_weights", True):
        paligemma_checkpoint_path = config["paligemma_checkpoint_path"]
        try:
            graphdef, state = nnx.split(model)
            params = state.to_pure_dict() 

            weight_loader = PaliGemmaWeightLoader(paligemma_checkpoint_path)
            loaded_params = weight_loader.load(params) 
            state = state.replace_by_pure_dict(loaded_params)
            model = nnx.merge(graphdef, state)
            logger.info("PaliGemma weights loaded successfully.")
        except Exception as e:
            logger.warning(f"Failed to load PaliGemma checkpoint: {e}. Using randomly initialized weights.")

    # Initialize tokenizer
    tokenizer = PaligemmaTokenizer(max_len=config["max_token_len"])

    # Create dataset
    dataset = RLDSParquetDataset(
        parquet_paths=config["data_paths"],
        image_size=config["image_size"],
        max_samples_per_file=config.get("max_samples_per_file", None),
        use_both_views=True,
        action_horizon=config["action_horizon"],
        tokenizer=tokenizer,
        max_token_len=config["max_token_len"],
    )

    # Define optimizer
    optimizer = optax.adamw(learning_rate=config["lr"])
    graphdef, state = nnx.split(model)
    trainable_params_state = state.to_pure_dict()
    opt_state = optimizer.init(trainable_params_state)

    def mse_loss(predictions, targets):
        return jnp.mean(jnp.square(predictions - targets))

    # @jax.jit
    def train_step(state_model, opt_state, batch, rng_key_step):
        graphdef, state = nnx.split(state_model)
        reconstructed_model = nnx.merge(graphdef, state)

        # ?? 注意：sample_actions 不参与反向传播！
        observation = Observation(
            images={k: jnp.asarray(v, dtype=jnp.float32) for k, v in batch["images"].items()},
            image_masks={k: jnp.asarray(v, dtype=jnp.bool_) for k, v in batch["image_masks"].items()},
            state=jnp.asarray(batch["state"], dtype=jnp.float32),
            tokenized_prompt=jnp.asarray(batch["tokenized_prompt"], dtype=jnp.int32),
            tokenized_prompt_mask=jnp.asarray(batch["tokenized_prompt_mask"], dtype=jnp.bool_),
        )
        predicted_actions = reconstructed_model.sample_actions(
            rng_key_step, observation, num_steps=config["action_horizon"]
        )

        def loss_fn(params):
            # 不再计算 predicted_actions，只用于监督损失
            return mse_loss(predicted_actions, batch["action"])

        loss_value, grads = jax.value_and_grad(loss_fn)(state.to_pure_dict())
        updates, new_opt_state = optimizer.update(grads, opt_state, state.to_pure_dict())
        new_params = optax.apply_updates(state.to_pure_dict(), updates)
        new_state_model = nnx.merge(graphdef, nnx.State(new_params))
        return new_state_model, new_opt_state, loss_value


    num_epochs = config.get("epochs", 10)
    batch_size = config.get("batch_size", 12)
    save_dir = config.get("save_dir", "checkpoints")
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(num_epochs):
        total_loss = 0.0
        num_batches = (len(dataset) + batch_size - 1) // batch_size
        indices = np.arange(len(dataset))
        np.random.shuffle(indices)

        with tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
            for i in pbar:
                batch_indices = indices[i * batch_size : (i + 1) * batch_size]
                batch_data = [dataset[idx] for idx in batch_indices]

                if not batch_data:
                    continue

                collated_batch = custom_collate_fn(batch_data)
                key, dropout_key_step = jax.random.split(key)
                model, opt_state, loss = train_step(model, opt_state, collated_batch, dropout_key_step)
                total_loss += loss.item()
                pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / num_batches
        logger.info(f"? [Epoch {epoch+1}] Avg Loss: {avg_loss:.4f}")
        checkpoint_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch+1}.msgpack")
        nnx.save(model, checkpoint_path) 
        logger.info(f"? Saved checkpoint to {checkpoint_path}")

    final_save_path = config.get("save_path", os.path.join(save_dir, "final_trained_pi0.msgpack"))
    nnx.save(model, final_save_path)
    logger.info(f"? Final model saved to {final_save_path}")


if __name__ == "__main__":
    import glob # Added for dataset file listing

    config = {
        "action_dim": 7,
        "action_horizon": 50, # Assuming Pi0 action horizon will be used for sampling actions
        "max_token_len": 48,
        "paligemma_variant": "gemma_2b",
        "action_expert_variant": "gemma_300m",
        "dtype": "bfloat16",
        "load_paligemma_weights": True,
        "paligemma_checkpoint_path": "./weight/pt_224.npz", # Adjust path as needed
        "data_paths": "/root/private_data/dataset/xarm6_parquet", # Example data path
        "image_size": 224,
        "lr": 1e-4,
        "epochs": 10,
        "batch_size": 12,
        "save_dir": "checkpoints_pi0",
        "save_path": "checkpoints_pi0/final_trained_pi0.msgpack",
        # "max_samples_per_file": 100, # Uncomment to limit samples per parquet file for testing
    }

    train(config)
