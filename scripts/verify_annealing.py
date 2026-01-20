import numpy as np

def calculate_beta(epoch, epochs, beta_kl, warmup_epochs, annealing_type):
    # Start annealing immediately, but from a very small value
    total_anneal_epochs = max(epochs // 2, 200)  # Longer annealing
    
    if epoch <= warmup_epochs:
        # Gradual warmup from 0 to small value (1% of target)
        return beta_kl * 0.01 * (epoch / warmup_epochs)
    
    # Continue annealing after warmup
    adjusted_epoch = epoch - warmup_epochs
    
    if annealing_type == 'sigmoid':
        # Smooth sigmoid curve
        k = 8 / total_anneal_epochs
        x0 = total_anneal_epochs / 2
        progress = 1 / (1 + np.exp(-k * (adjusted_epoch - x0)))
        
        # Scale to reach config.beta_kl gradually
        beta = beta_kl * progress * 0.3  # Max 30% of target beta
    else:
        # Linear annealing
        progress = min(1.0, adjusted_epoch / total_anneal_epochs)
        beta = beta_kl * progress * 0.3
    
    return beta

def verify_annealing():
    epochs = 1000
    beta_kl = 0.001
    warmup = 50
    
    print(f"Annealing Verification (Epochs={epochs}, Target Beta={beta_kl}, Warmup={warmup})")
    print("-" * 60)
    print(f"{'Epoch':>6} | {'Beta (Sigmoid)':>15} | {'Beta (Linear)':>15}")
    print("-" * 60)
    
    for e in [1, 10, 50, 100, 200, 500, 1000]:
        beta_sig = calculate_beta(e, epochs, beta_kl, warmup, 'sigmoid')
        beta_lin = calculate_beta(e, epochs, beta_kl, warmup, 'linear')
        print(f"{e:6d} | {beta_sig:15.8f} | {beta_lin:15.8f}")

if __name__ == "__main__":
    verify_annealing()
