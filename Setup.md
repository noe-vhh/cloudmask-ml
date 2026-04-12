# CloudMask ML - Environment Setup

Reproducibility guide for the CloudMask semantic segmentation prototype.
Covers GPU compute stack and Python environment setup from a clean Linux install.

---

## Target Hardware & OS

| Component | Spec |
|-----------|------|
| GPU | AMD Radeon RX 6800 XT (Navi 21, gfx1030, 16GB VRAM) |
| OS | Ubuntu 24.04 LTS (Noble) |
| Kernel | 6.17+ |
| ROCm | 6.4 |
| PyTorch | 2.x (ROCm 6.2 wheel) |

> **Note:** ROCm 6.2 PyTorch wheels are used on a ROCm 6.4 install -
> the HIP runtime is backwards compatible.
> Other AMD RDNA2/RDNA3 GPUs (gfx1030, gfx1100 family) should work with
> minimal changes. NVIDIA GPU users should install the standard CUDA PyTorch
> wheel instead and can skip the ROCm steps entirely.

---

## Prerequisites

- Ubuntu 24.04 LTS installed and booted
- User has `sudo` privileges
- Internet access
- AMD GPU physically installed and visible to OS:

```bash
lspci | grep -i "vga\|amd"
lsmod | grep amdgpu   # should return output - driver is built into kernel
```

---

## 1. Add User to GPU Access Groups

ROCm requires your user to be in the `render` and `video` groups to access
the GPU without root privileges.

```bash
sudo usermod -aG render,video $USER
```

Log out and back in (or reboot) for group changes to take effect. Verify with:

```bash
groups | grep -E "render|video"
```

---

## 2. Add AMD ROCm apt Repository

```bash
# Install dependencies
sudo apt install -y wget gnupg2

# Add AMD's GPG signing key
wget https://repo.radeon.com/rocm/rocm.gpg.key -O - | \
  gpg --dearmor | \
  sudo tee /etc/apt/keyrings/rocm.gpg > /dev/null

# Add the ROCm 6.4 repository for Ubuntu 24.04
echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] \
  https://repo.radeon.com/rocm/apt/6.4 noble main" | \
  sudo tee /etc/apt/sources.list.d/rocm.list

sudo apt update
```

---

## 3. Pin ROCm Packages

Prevents apt from mixing AMD's ROCm packages with older versions
from Ubuntu's standard repositories, which causes dependency conflicts.

```bash
sudo tee /etc/apt/preferences.d/rocm-pin << 'EOF'
Package: *
Pin: release o=repo.radeon.com
Pin-Priority: 1001
EOF
```

---

## 4. Install ROCm Compute Stack

We install individual components rather than the `rocm-hip-sdk` meta-package
to avoid pulling in `rccl` (multi-GPU communications library), which has an
unresolvable dependency (`libdrm-amdgpu-amdgpu1`) on Ubuntu 24.04 with
kernel 6.17. Single-GPU setups do not need `rccl`.

```bash
sudo apt install -y \
  rocm-hip-runtime \
  rocm-hip-runtime-dev \
  rocm-opencl-runtime \
  rocminfo \
  rocm-smi-lib \
  hipcc \
  hip-dev
```

---

## 5. Configure Shell Environment

```bash
echo 'export PATH=$PATH:/opt/rocm/bin' >> ~/.zshrc        # or ~/.bashrc
echo 'export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH' >> ~/.zshrc

source ~/.zshrc
```

Then reboot:

```bash
sudo reboot
```

---

## 6. Verify ROCm

After reboot:

```bash
rocm-smi
```

Expected output: a table showing your GPU with temperature, power draw,
and VRAM usage. If your card appears, ROCm is working correctly.

---

## 7. Set Up Project & Virtual Environment

```bash
# Create project directory
mkdir -p ~/projects/cloudmask
cd ~/projects/cloudmask

# Initialise Git
git init
git branch -M main

# Install venv support if not present (Ubuntu splits this out)
sudo apt install -y python3.12-venv

# Create venv inside project - named .venv (hidden, industry convention)
python3 -m venv .venv

# Activate
source .venv/bin/activate
```

Create a `.gitignore` to prevent the venv and build artifacts from being committed:

```bash
cat > .gitignore << 'EOF'
.venv/
__pycache__/
*.pyc
*.pyo
.DS_Store
*.egg-info/
dist/
build/
EOF
```

> **Why not commit the venv?** It contains ~4GB of OS-specific binaries that
> are useless on another machine. Git tracks source code - dependencies are
> recreated from `requirements.txt` instead.

---

## 8. Install PyTorch (ROCm build)

```bash
pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/rocm6.2
```

This downloads ~2.5GB. The `rocm6.2` index provides PyTorch wheels built
against ROCm's HIP runtime, which is what enables GPU acceleration on AMD.

---

## 9. Verify PyTorch GPU Detection

```bash
python3 -c "
import torch
print('PyTorch version:', torch.__version__)
print('ROCm available:', torch.cuda.is_available())
print('GPU name:', torch.cuda.get_device_name(0))
print('VRAM:', round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1), 'GB')
"
```

Expected output:
```
PyTorch version: 2.x.x+rocm6.2
ROCm available: True
GPU name: AMD Radeon RX 6800 XT
VRAM: 16.0 GB
```

> **Why does PyTorch use `.cuda()` for an AMD GPU?**
> ROCm implements AMD's HIP API as a drop-in replacement for CUDA.
> PyTorch exposes both through the same `.cuda()` interface for compatibility.
> `torch.cuda.is_available()` returning `True` on AMD + ROCm is correct.

---

## Activation Quick Reference

Each new session (from project root):

```bash
cd ~/projects/cloudmask
source .venv/bin/activate
```

To deactivate:

```bash
deactivate
```

---

## Troubleshooting

**`rocm-smi` not found after install**
Ensure `/opt/rocm/bin` is on your PATH and you've sourced your shell config.

**`torch.cuda.is_available()` returns `False`**
- Confirm `rocm-smi` shows your GPU
- Confirm you installed the `rocm6.2` wheel, not the default PyTorch wheel
- Confirm your user is in the `render` and `video` groups

**Dependency conflict on `rocm-hip-sdk` install**
Do not install `rocm-hip-sdk` directly - use the individual package list
in Step 4 above. The `rccl` package it pulls in is not installable on
Ubuntu 24.04 + kernel 6.17 due to a missing patched `libdrm` variant.