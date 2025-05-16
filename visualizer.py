# -*- coding: utf-8 -*-
# Copyright © 2025 Joshuah Rainstar
# License: see LICENSE.txt

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import ipywidgets as widgets
from IPython.display import display, clear_output
import time

# --- Configuration Constants ---
CHAR_WIDTH = 8  # Font size 8 for token rendering
CHAR_HEIGHT = 11
SEQ_LEN = 128
BATCH_SIZE = 16
LOSS_BAR_HEIGHT = 32
EWMA_HEIGHT = 32  # Increased to accommodate large text (previously 32)

# Full-resolution framebuffer dimensions
container_width = CHAR_WIDTH * SEQ_LEN  # 1024 pixels
container_height = CHAR_HEIGHT * BATCH_SIZE  # 176 pixels
total_height = container_height + LOSS_BAR_HEIGHT + EWMA_HEIGHT  # Adjusted for larger EWMA

# Final scaled-down dimensions
scaled_width = container_width   # 512 pixels
scaled_height = total_height  # 170 pixels

# Initialize framebuffer
framebuffer = np.zeros((total_height, container_width, 3), dtype=np.uint8)

# EWMA storage
ticker_history = np.zeros(SEQ_LEN, dtype=np.float32)  # Stock ticker moving buffer
loss_memory = 0.0
# Load font
try:
    font = ImageFont.truetype("DejaVuSansMono.ttf", 8)  # Monospaced font
    font_large = ImageFont.truetype("DejaVuSansMono.ttf", 64)  # Large EWMA display
except:
    font = ImageFont.load_default()
    font_large = font

# --- Color Mapping Functions ---
def get_flame_color(val):
    """Map a normalized value to a flame-like color."""
    return np.array([int(val * 255), int(val * 0.5 * 255), 0], dtype=np.uint8)

# --- IPython Display Setup ---
out = widgets.Output()
display(out)

def get_dynamic_color(attn_val, loss_val):
    """
    Compute a dynamic color transition between flame orange (uncertain) and phosphor green (confident).

    attn_val: Normalized attention value (0 to 1)
    loss_val: Normalized loss value (0 to 1, inverted as 1 - loss)

    Returns an RGB color as a NumPy array.
    colors late in training will often be red. this is suggested to swap out for get_flame_color
    but only on fine tuning on new data.
    """
    certainty = 1 - loss_val  # High certainty = low loss

    # Define RGB endpoints
    orange = np.array([attn_val * 255, attn_val * 0.5 * 255, 0], dtype=np.uint8)   # Uncertain (High Loss)
    green = np.array([attn_val * 0.5 * 255, attn_val * 255, attn_val * 0.25 * 255], dtype=np.uint8)  # Confident (Low Loss)

    # Interpolate based on certainty (0 = uncertain/orange, 1 = confident/green)
    color = (certainty * green) + ((1 - certainty) * orange)

    return color.astype(np.uint8)
def normalize_rows(x: np.ndarray) -> np.ndarray:
    min_val = np.min(x, axis=1, keepdims=True)
    max_val = np.max(x, axis=1, keepdims=True)
    scale = max_val - min_val
    return (x - min_val) / (scale + 1e-16)
    
# --- Framebuffer Update Function ---
def update_framebuffer(attn_weights, token_losses, current_loss, tokens):
    token_losses = normalize_rows(token_losses)
    attn_weights = normalize_rows(attn_weights)
    """Render the text grid with coloration based on attn * inverse loss."""
    global framebuffer, loss_history, ticker_history, loss_memory

    # Normalize to [0,1]

    # Create image buffer
    img = Image.new("RGB", (container_width, total_height), (0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Render text with colored intensity
    char_positions = [
        (col * CHAR_WIDTH, row * CHAR_HEIGHT + EWMA_HEIGHT + LOSS_BAR_HEIGHT, tokens[row][col])
        for row in range(BATCH_SIZE) for col in range(SEQ_LEN)
    ]
    colors = [
        tuple(get_dynamic_color(attn_weights[row, col], token_losses[row, col]))
        for row in range(BATCH_SIZE) for col in range(SEQ_LEN)
    ]
    for (x, y, char), color in zip(char_positions, colors):
        draw.text((x, y), char, font=font, fill=color)


    etcerta = 0.367879441  # Constant used in update rule
    et = 1 - etcerta
    update = loss_memory * et + np.minimum(12, np.maximum(current_loss , 0)) * etcerta
    loss_memory = loss_memory * et + update * etcerta
    # --- EWMA Display (LARGE FONT) ---
    ewma = loss_memory
    ewma_text = f"{ewma:.4f}"
    draw.text((container_width-128, 0), ewma_text, font_size=32, fill=(65,255, 125))

    # --- Moving Loss Ticker Graph ---
    ticker_history = np.roll(ticker_history, -1)  # Shift left
    ticker_history[-1] = current_loss  # Insert new loss on the right

    # Rescale ticker dynamically like a stock ticker (normalize to min-max range)
    min_loss = np.min(ticker_history)
    max_loss = np.max(ticker_history)
    range_loss = max_loss - min_loss if max_loss != min_loss else 1
    normalized_ticker = (ticker_history - min_loss) / range_loss

    # Draw ticker graph line
    # Optimized drawing loop (fewer function calls)
    y_vals = EWMA_HEIGHT + (1 - normalized_ticker) * LOSS_BAR_HEIGHT
    x_vals = np.arange(SEQ_LEN) * CHAR_WIDTH
    for i in range(SEQ_LEN - 1):
        draw.line([(x_vals[i], y_vals[i]), (x_vals[i + 1], y_vals[i + 1])], fill=(0, 255, 255), width=2)

    framebuffer = np.array(img)

# --- IPython Display Update Function ---
def update_display():
    """Show the framebuffer, scaled down by half using ipywidgets."""
    img = Image.fromarray(framebuffer)
    img_resized = img.resize((scaled_width, scaled_height), Image.LANCZOS)

    with out:
        clear_output(wait=True)
        display(img_resized)

loss_history = []

'''
use like this:
deep in your model's attention:
 attn_score = weights.sum(dim=2)  # (B, K)

    # Normalize each sample separately (min-max per row)
    min_vals = attn_score.min(dim=-1, keepdim=True).values
    max_vals = attn_score.max(dim=-1, keepdim=True).values
    attn_score = (attn_score - min_vals) / (max_vals - min_vals + 1e-6)  # (B, K)

then in training:
per_token_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            yb.view(-1),
            reduction='none'  # This gives the raw loss per token
        ).reshape(B,T) # Shape: (B*S,)
        loss = per_token_loss.mean() + kl *0.1
        loss_cpu = per_token_loss.cpu().detach().numpy()
        tokens = [[itos[idx] for idx in seq.tolist()] for seq in yb]
        attn_cpu = attn_weights.cpu().detach().numpy()
        update_framebuffer(attn_cpu, loss_cpu, loss.item(), tokens)
        update_display()

'''
