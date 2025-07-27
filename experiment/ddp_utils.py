#!/usr/bin/env python3
"""
DDP (DistributedDataParallel) Utilities for Model Loading and Saving
Shared utilities to handle DDP prefix issues across training scripts
"""

import os
import sys
import tempfile
import shutil
import glob
import logging
from typing import Union, Type
import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    PreTrainedModel
)
from safetensors import safe_open
from safetensors.torch import save_file

def setup_ddp_logging():
    """Setup logging for DDP utilities"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def load_ddp_model_correctly(
    model_path: str, 
    model_class: Type[PreTrainedModel] = AutoModelForCausalLM,
    **model_kwargs
) -> PreTrainedModel:
    """
    Load a model that was saved with DistributedDataParallel (DDP) format.
    This handles the 'model.module.*' prefix issue by creating corrected weight files.
    
    Args:
        model_path: Path to the model directory
        model_class: Model class to instantiate (AutoModelForCausalLM, AutoModelForSequenceClassification, etc.)
        **model_kwargs: Additional arguments to pass to model.from_pretrained()
    
    Returns:
        Loaded model with corrected weights
    """
    logger = setup_ddp_logging()
    
    # Set default kwargs if not provided
    default_kwargs = {
        "device_map": "auto",
        "torch_dtype": torch.float16,
        "trust_remote_code": True
    }
    default_kwargs.update(model_kwargs)
    
    try:
        # Check if this is a DDP checkpoint by looking at safetensors files
        safetensor_files = glob.glob(os.path.join(model_path, "*.safetensors"))
        if not safetensor_files:
            logger.info("No safetensors files found, using standard loading")
            return model_class.from_pretrained(model_path, **default_kwargs)
        
        # Check if this is actually a DDP checkpoint by inspecting keys
        is_ddp_checkpoint = False
        for safetensor_file in safetensor_files:
            with safe_open(safetensor_file, framework="pt", device="cpu") as f:
                keys = list(f.keys())
                if any(key.startswith("model.module.") or key.startswith("module.") for key in keys):
                    is_ddp_checkpoint = True
                    break
        
        if not is_ddp_checkpoint:
            logger.info("ℹ️ No DDP format detected, using standard loading")
            return model_class.from_pretrained(model_path, **default_kwargs)
        
        logger.info("🔧 DDP format detected, creating corrected model checkpoint...")
        
        # Create a temporary directory for corrected weights
        with tempfile.TemporaryDirectory() as temp_dir:
            # Copy all non-safetensors files to temp directory
            for file_name in os.listdir(model_path):
                if not file_name.endswith('.safetensors'):
                    src_path = os.path.join(model_path, file_name)
                    if os.path.isfile(src_path):
                        shutil.copy2(src_path, temp_dir)
            
            # Process and save corrected weights
            logger.info(f"Processing {len(safetensor_files)} safetensor files...")
            for i, safetensor_file in enumerate(safetensor_files):
                logger.info(f"  Processing {os.path.basename(safetensor_file)}...")
                
                corrected_state_dict = {}
                with safe_open(safetensor_file, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        tensor = f.get_tensor(key)
                        # Remove DDP prefix - handle both "model.module." and "module."
                        clean_key = key
                        if key.startswith("model.module."):
                            clean_key = key[13:]  # Remove "model.module."
                        elif key.startswith("module."):
                            clean_key = key[7:]   # Remove "module."
                        
                        corrected_state_dict[clean_key] = tensor
                
                # Save corrected weights with proper naming
                if len(safetensor_files) == 1:
                    output_file = os.path.join(temp_dir, "model.safetensors")
                else:
                    output_file = os.path.join(temp_dir, f"model-{i+1:05d}-of-{len(safetensor_files):05d}.safetensors")
                
                save_file(corrected_state_dict, output_file)
                logger.info(f"    ✅ Saved corrected weights to {os.path.basename(output_file)}")
            
            # Now load the model from the corrected directory
            logger.info("🔄 Loading model with corrected weights...")
            model = model_class.from_pretrained(temp_dir, **default_kwargs)
            
            logger.info("✅ DDP model loaded successfully with corrected weights")
            return model
                
    except Exception as e:
        logger.error(f"DDP loading failed: {e}")
        logger.info("Falling back to standard loading...")
        return model_class.from_pretrained(model_path, **default_kwargs)

def save_model_without_ddp_prefix(
    model: PreTrainedModel,
    output_dir: str,
    tokenizer=None,
    safe_serialization: bool = True
):
    """
    Save model ensuring no DDP prefixes in the saved weights.
    
    Args:
        model: Model to save
        output_dir: Directory to save the model
        tokenizer: Optional tokenizer to save alongside
        safe_serialization: Whether to use safetensors format
    """
    logger = setup_ddp_logging()
    
    try:
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Get the unwrapped model (remove DDP wrapper if present)
        unwrapped_model = model
        if hasattr(model, 'module'):
            unwrapped_model = model.module
            logger.info("🔧 Unwrapping DDP model for saving")
        
        # Handle shared tensor issues for safetensors
        if safe_serialization:
            try:
                # Check if lm_head and embeddings share memory (common issue)
                if (hasattr(unwrapped_model, 'lm_head') and 
                    hasattr(unwrapped_model, 'model') and 
                    hasattr(unwrapped_model.model, 'embed_tokens')):
                    
                    lm_head_ptr = unwrapped_model.lm_head.weight.data_ptr()
                    embed_ptr = unwrapped_model.model.embed_tokens.weight.data_ptr()
                    
                    if lm_head_ptr == embed_ptr:
                        logger.info("🔧 Fixing shared memory between lm_head and embed_tokens...")
                        unwrapped_model.lm_head.weight = nn.Parameter(
                            unwrapped_model.lm_head.weight.clone()
                        )
                        logger.info("✅ Tensor sharing resolved")
                
                # Try saving with safetensors
                unwrapped_model.save_pretrained(
                    output_dir,
                    safe_serialization=True,
                    max_shard_size="5GB"
                )
                logger.info("✅ Model saved with safetensors format")
                
            except RuntimeError as e:
                if "Some tensors share memory" in str(e):
                    logger.warning("⚠️ Safetensors sharing issue, falling back to PyTorch format")
                    # Fallback to PyTorch format
                    unwrapped_model.save_pretrained(
                        output_dir,
                        safe_serialization=False,
                        max_shard_size="5GB"
                    )
                    logger.info("✅ Model saved with PyTorch format (fallback)")
                else:
                    raise
        else:
            # Use PyTorch format directly
            unwrapped_model.save_pretrained(
                output_dir,
                safe_serialization=False,
                max_shard_size="5GB"
            )
            logger.info("✅ Model saved with PyTorch format")
        
        # Save tokenizer if provided
        if tokenizer is not None:
            tokenizer.save_pretrained(output_dir)
            logger.info("✅ Tokenizer saved")
            
    except Exception as e:
        logger.error(f"❌ Failed to save model: {e}")
        raise

def check_ddp_model_format(model_path: str) -> bool:
    """
    Check if a model directory contains DDP-formatted weights.
    
    Args:
        model_path: Path to model directory
        
    Returns:
        True if DDP format detected, False otherwise
    """
    safetensor_files = glob.glob(os.path.join(model_path, "*.safetensors"))
    if not safetensor_files:
        return False
    
    for safetensor_file in safetensor_files:
        try:
            with safe_open(safetensor_file, framework="pt", device="cpu") as f:
                keys = list(f.keys())
                if any(key.startswith("model.module.") or key.startswith("module.") for key in keys):
                    return True
        except Exception:
            continue
    
    return False

def convert_ddp_checkpoint_inplace(model_path: str) -> bool:
    """
    Convert DDP checkpoint in place, replacing the original files.
    
    Args:
        model_path: Path to model directory to convert
        
    Returns:
        True if conversion was performed, False if no conversion needed
    """
    logger = setup_ddp_logging()
    
    if not check_ddp_model_format(model_path):
        logger.info("No DDP format detected, no conversion needed")
        return False
    
    logger.info(f"🔧 Converting DDP checkpoint in place: {model_path}")
    
    # Create backup directory
    backup_dir = f"{model_path}_ddp_backup"
    if os.path.exists(backup_dir):
        shutil.rmtree(backup_dir)
    shutil.copytree(model_path, backup_dir)
    logger.info(f"📁 Created backup at: {backup_dir}")
    
    try:
        # Process safetensors files
        safetensor_files = glob.glob(os.path.join(model_path, "*.safetensors"))
        
        for i, safetensor_file in enumerate(safetensor_files):
            logger.info(f"  Processing {os.path.basename(safetensor_file)}...")
            
            corrected_state_dict = {}
            with safe_open(safetensor_file, framework="pt", device="cpu") as f:
                for key in f.keys():
                    tensor = f.get_tensor(key)
                    # Remove DDP prefix
                    clean_key = key
                    if key.startswith("model.module."):
                        clean_key = key[13:]
                    elif key.startswith("module."):
                        clean_key = key[7:]
                    
                    corrected_state_dict[clean_key] = tensor
            
            # Save corrected weights, overwriting original
            save_file(corrected_state_dict, safetensor_file)
            logger.info(f"    ✅ Converted {os.path.basename(safetensor_file)}")
        
        logger.info("✅ DDP checkpoint converted successfully")
        logger.info(f"📁 Original checkpoint backed up to: {backup_dir}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Conversion failed: {e}")
        # Restore from backup
        shutil.rmtree(model_path)
        shutil.move(backup_dir, model_path)
        logger.info("🔄 Restored original checkpoint from backup")
        raise
