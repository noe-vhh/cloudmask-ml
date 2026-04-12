# tmux Quick Reference

tmux is a terminal multiplexer - it runs terminal sessions on the remote machine
itself, independent of your SSH connection. If your laptop sleeps or disconnects,
everything keeps running.

---

## Essential Workflow

### Start a named session
```bash
tmux new -s session_name
```
Example: `tmux new -s cloudmask`

### Detach (leave running, go back to normal terminal)
```
Ctrl+B  then  D
```
The session keeps running on the rig. Safe to sleep your laptop.

### List running sessions
```bash
tmux ls
```

### Reattach to a session
```bash
tmux attach -t session_name
```
Example: `tmux attach -t cloudmask`

### Kill a session (when you're done)
```bash
tmux kill-session -t session_name
```

---

## Inside tmux - Window & Pane Management

All tmux commands start with the **prefix**: `Ctrl+B`

### Windows (like browser tabs)
| Action | Keys |
|--------|------|
| New window | `Ctrl+B  C` |
| Next window | `Ctrl+B  N` |
| Previous window | `Ctrl+B  P` |
| List windows | `Ctrl+B  W` |
| Rename window | `Ctrl+B  ,` |

### Panes (split the terminal)
| Action | Keys |
|--------|------|
| Split horizontal | `Ctrl+B  %` |
| Split vertical | `Ctrl+B  "` |
| Navigate panes | `Ctrl+B  Arrow keys` |
| Close pane | `Ctrl+B  X` |
| Zoom pane (fullscreen toggle) | `Ctrl+B  Z` |

---

## Scrolling in tmux

By default you can't scroll with the mouse. Enter scroll mode first:
```
Ctrl+B  [
```
Then use arrow keys or Page Up/Down to scroll.
Press `Q` to exit scroll mode.

---

## Typical CloudMask Workflow

```bash
# On the rig - start a session for a long job
tmux new -s training

# Inside tmux - run your job
cd ~/projects/cloudmask
source .venv/bin/activate
python src/train.py --config config.yaml

# Detach and go do something else
Ctrl+B  D

# Later - check on it
tmux attach -t training
```

---

## Pro Tips

- **Name your sessions** - `tmux new -s training` not just `tmux new`. You'll thank yourself when you have 3 running.
- **One session per long job** - download, training, evaluation each get their own session.
- **`tmux ls` first** - always check what's running before starting a new session.
- **Don't nest tmux** - if you SSH into a machine that already has tmux running and start another tmux inside it, you'll confuse yourself. Check with `tmux ls` first.

---

## If tmux isn't installed
```bash
sudo apt install -y tmux
```