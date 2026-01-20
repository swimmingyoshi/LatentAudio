#!/usr/bin/env python3
"""Quick training health check - run with: python scripts/check_health.py"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from latentaudio.core.training import calculate_beta
from latentaudio.types import TrainingConfig

def main():
    log_dir = Path('logs')
    if not log_dir.exists():
        print("No logs directory found")
        return
    
    folders = sorted([f for f in log_dir.iterdir() if f.is_dir()])
    if not folders:
        print("No training logs found")
        return
    
    log_path = folders[-1]
    events = list(log_path.glob('events.out.tfevents.*'))
    if not events:
        print("No event files found")
        return
    
    # Read events
    from tensorboard.backend.event_processing import event_accumulator
    ea = event_accumulator.EventAccumulator(str(events[-1]), size_guidance={'scalars': 10000})
    ea.Reload()
    
    # Get available keys
    available_keys = set(ea.scalars.Keys())
    
    # Get data
    def get_latest(key):
        if key not in available_keys:
            return None
        vals = list(ea.Scalars(key))
        return vals[-1] if vals else None
    
    total = get_latest('Loss/Total')
    recon = get_latest('Loss/Reconstruction_Combined')
    kl = get_latest('Loss/KL')
    lr = get_latest('Training/Learning_Rate')
    
    if not total:
        print("No scalar data found")
        return
    
    epoch = total.step
    config = TrainingConfig()
    beta = calculate_beta(epoch, config)
    beta_pct = (beta / config.beta_kl * 100) if config.beta_kl > 0 else 0
    
    # Get first value for comparison
    first_total = list(ea.Scalars('Loss/Total'))[0] if 'Loss/Total' in available_keys else total
    loss_change = ((total.value - first_total.value) / first_total.value * 100) if first_total.value > 0 else 0
    
    print()
    print("=" * 50)
    print("TRAINING HEALTH CHECK")
    print("=" * 50)
    print(f"Log:     {log_path.name}")
    print(f"Epoch:   {epoch}")
    print(f"LR:      {lr.value:.6f}")
    print(f"Beta:    {beta:.6f} ({beta_pct:.0f}% of max)")
    print()
    print(f"Total:   {total.value:.4f}")
    print(f"Recon:   {recon.value:.4f}")
    print(f"KL:      {kl.value:.4f}")
    print()
    
    # Status
    if loss_change < -20:
        print("Status:  [OK] Loss decreasing")
    elif loss_change < 0:
        print("Status:  [--] Loss slowly dropping")
    else:
        print("Status:  [!!] Loss not improving")
    
    if kl.value < 10:
        print("         [!!] KL too low (collapse risk)")
    elif kl.value > 100:
        print("         [!!] KL high (unstable)")
    elif 40 < kl.value < 80:
        print("         [OK] KL healthy")
    else:
        print("         [--] KL moderate")
    
    if beta_pct < 50:
        print("         [--] Beta warming up")
    else:
        print("         [OK] Beta active")
    
    print()
    print(f"Change:  {loss_change:+.1f}% since start")
    print()
    
    if kl.value > 100:
        print("Rec:     KL high - increase beta_kl next run")
    elif loss_change > -10:
        print("Rec:     Loss stuck - check audio diversity")
    else:
        print("Rec:     Training healthy - continue or save")
    
    print("=" * 50)
    print()

if __name__ == '__main__':
    main()
