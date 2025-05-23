import dataclasses
import logging
import os
import re
from typing import Protocol, runtime_checkable

import flax.nnx as nnx
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image

from models import Pi0, Pi0Config
from models import Observation, Actions
from models import PaligemmaTokenizer
from shared import array_typing as at
import shared.download as download

# from tokenizer import PaligemmaTokenizer # Import PaligemmaTokenizer

# 设置日志级别
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@runtime_checkable
class WeightLoader(Protocol):
    def load(self, params: at.Params) -> at.Params:
        """加载模型权重。

        Args:
            params: 模型的参数。这是一个嵌套的类数组对象结构，表示模型的参数。

        Returns:
            加载的参数。结构必须与`params`相同。如果返回参数的子集，加载器必须将加载的参数与`params`合并。
        """
        ...


@dataclasses.dataclass(frozen=True)
class PaliGemmaWeightLoader(WeightLoader):
    """从官方PaliGemma检查点加载权重。

    这将覆盖具有相似名称的现有权重，同时保持所有额外的权重不变。
    这允许我们支持Pi0模型使用的动作专家。
    """

    params_path: str

    def load(self, params: at.Params) -> at.Params:
        path = download.maybe_download(
            self.params_path, gs={"token": "anon"}
        )
        with path.open("rb") as f:
            # Load flat parameters from the .npz file
            flat_params_np = dict(np.load(f, allow_pickle=False))

        # The .npz file keys start with "params/...", so unflatten to get the
        # structure like {'params': {'img': {...}, 'llm': {...}}}
        unflattened_all_params = flax.traverse_util.unflatten_dict(flat_params_np, sep="/")
        
        # Ensure 'params' key exists as expected from the checkpoint structure
        if "params" not in unflattened_all_params:
            raise ValueError(
                f"Expected 'params' key in loaded checkpoint, but found: {unflattened_all_params.keys()}"
            )

        # Wrap the actual loaded parameters under a 'PaliGemma' key to match 
        # the Pi0 model's internal parameter structure for the PaliGemma component.
        loaded_params_for_merge = {"PaliGemma": unflattened_all_params["params"]}
        
        # Merge the loaded parameters with the existing model parameters.
        # This function now ensures all values are JAX arrays.
        return _merge_params(loaded_params_for_merge, params, missing_regex=".*")


def _merge_params(loaded_params: at.Params, params: at.Params, *, missing_regex: str) -> at.Params:
    """将加载的参数与引用参数合并。

    Args:
        loaded_params: 要合并的参数。
        params: 引用参数。
        missing_regex: 应该从引用参数合并的所有缺失键的正则表达式模式。

    Returns:
        一个包含合并参数的新字典，所有叶节点都为JAX数组。
    """
    flat_ref = flax.traverse_util.flatten_dict(params, sep="/")
    flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")

    result = {}
    # First, process and convert all loaded parameters
    for k, v in flat_loaded.items():
        if k in flat_ref:
            # Convert the loaded NumPy array 'v' to a JAX array with the target dtype
            result[k] = jnp.asarray(v, dtype=flat_ref[k].dtype)
        else:
            # Log a warning if a key from the loaded checkpoint doesn't exist in the model's parameters
            logger.warning(
                f"Loaded parameter key '{k}' not found in reference model parameters. Skipping."
            )

    # Then, merge any missing parameters from the reference model based on the regex
    pattern = re.compile(missing_regex)
    for k in {k for k in flat_ref if pattern.fullmatch(k)}:
        if k not in result:
            # Ensure parameters from the reference model are also JAX arrays
            # (they typically already are, but jnp.asarray is safe for consistency)
            result[k] = jnp.asarray(flat_ref[k])

    # Unflatten the result back into a nested dictionary structure
    return flax.traverse_util.unflatten_dict(result, sep="/")


def load_image(image_path, target_size=(224, 224)):
    """加载并预处理图像

    Args:
        image_path: 图像文件路径
        target_size: 目标尺寸，默认为(224, 224)

    Returns:
        预处理后的图像数组，范围为[-1, 1]
    """
    image = Image.open(image_path).convert("RGB")
    image = image.resize(target_size, Image.LANCZOS)
    image_array = np.array(image).astype(np.float32) / 127.5 - 1.0
    return image_array


def main():
    # 初始化随机数生成器
    rng = jax.random.key(0)

    # 创建Pi0模型配置
    config = Pi0Config(
        action_dim=7,
        action_horizon=50,
        max_token_len=48,
        paligemma_variant="gemma_2b",
        action_expert_variant="gemma_300m",
        dtype="bfloat16"
    )

    logger.info("Creating Pi0 model...")
    # 创建Pi0模型
    model = config.create(rng)

    # Assumed PaliGemma pre-trained weights path
    paligemma_checkpoint_path = "./weight/pt_224.npz"

    # Load weights if the checkpoint file exists
    try:
        # Get the model's initial parameters as a pure dictionary of JAX arrays
        graphdef, state = nnx.split(model)
        params = state.to_pure_dict() 

        # Load and merge PaliGemma pre-trained weights
        weight_loader = PaliGemmaWeightLoader(paligemma_checkpoint_path)
        loaded_params = weight_loader.load(params) 
        
        # Log keys of loaded and merged parameters for verification
        print("Keys of loaded and merged parameters:")
        for k in flax.traverse_util.flatten_dict(loaded_params, sep="/").keys():
            print(f"- {k}")

        # Update model parameters with the loaded (and merged) JAX arrays
        state.replace_by_pure_dict(loaded_params)
        model = nnx.merge(graphdef, state)
        logger.info("PaliGemma weights loaded successfully")
    except Exception as e:
        logger.warning(f"Failed to load PaliGemma checkpoint: {e}")
        logger.info("Using randomly initialized weights")

    # Prepare model input
    # 1. Load example images
    # Assume there are images from three camera views
    image_paths = {
        "base_0_rgb": "path/to/base_camera_image.jpg",
        "wrist_0_rgb": "path/to/wrist_camera_image.jpg",
    }

    # Create a batch of size 1
    batch_size = 1

    # If image files exist, load images, otherwise use random images
    images = {}
    image_masks = {}
    for name, path in image_paths.items():
        if os.path.exists(path):  # Check if the file actually exists
            try:
                image = load_image(path)
                image = jnp.expand_dims(image, axis=0)  # Add batch dimension
                images[name] = image
                image_masks[name] = jnp.ones((batch_size,), dtype=jnp.bool_)
            except Exception as e:
                logger.warning(
                    f"Error loading image {path}: {e}, using random image instead."
                )
                images[name] = jax.random.uniform(
                    jax.random.key(0), shape=(batch_size, 224, 224, 3), minval=-1, maxval=1
                )
                image_masks[name] = jnp.ones((batch_size,), dtype=jnp.bool_)
        else:
            logger.warning(f"Image not found at {path}, using random image")
            images[name] = jax.random.uniform(
                jax.random.key(0), shape=(batch_size, 224, 224, 3), minval=-1, maxval=1
            )
            image_masks[name] = jnp.ones((batch_size,), dtype=jnp.bool_)

    # 2. Create state vector (robot's current state)
    state_vector = jnp.zeros((batch_size, config.action_dim), dtype=jnp.float32)

    # 3. Create text prompt and tokenize it using PaligemmaTokenizer
    prompt_text = "What is in the image?" # Example prompt
    tokenizer = PaligemmaTokenizer(max_len=config.max_token_len)
    tokens, mask = tokenizer.tokenize(prompt_text)

    # Add batch dimension to tokens and mask
    tokenized_prompt = jnp.expand_dims(tokens, axis=0)
    tokenized_prompt_mask = jnp.expand_dims(mask, axis=0)

    # Create observation object
    observation = Observation(
        images=images,
        image_masks=image_masks,
        state=state_vector,
        tokenized_prompt=tokenized_prompt,
        tokenized_prompt_mask=tokenized_prompt_mask
    )

    # Sample actions from the model
    logger.info("Sampling actions from the model...")
    actions = model.sample_actions(rng, observation, num_steps=10)

    print(f"Generated actions shape: {actions.shape}")
    print(f"First action: {actions[0, 0]}")


if __name__ == "__main__":
    main()