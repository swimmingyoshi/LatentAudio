import os
import glob
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import datetime

def get_latest_log_dir(logs_root='logs'):
    # Find all training directories
    dirs = [d for d in glob.glob(os.path.join(logs_root, 'training_*')) if os.path.isdir(d)]
    if not dirs:
        return None
    # Sort by modification time to get the most recent
    return max(dirs, key=os.path.getmtime)

def summarize_logs(log_dir, output_file='TRAINING_SUMMARY.md'):
    if not log_dir:
        print("No log directories found.")
        return

    print(f"Summarizing logs from: {log_dir}")
    ea = EventAccumulator(log_dir)
    ea.Reload()

    tags = ea.Tags()
    scalars = tags.get('scalars', [])
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"# Training Summary: {os.path.basename(log_dir)}\n\n")
        f.write(f"Generated at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # Get latest progress if available in text tags (tensors in modern TB)
        text_tags = tags.get('tensors', [])
        
        # Check for common text summary locations
        progress_text = None
        for tag in ['Training/Progress/text_summary', 'Training/Progress']:
            if tag in text_tags:
                progress_events = ea.Tensors(tag)
                if progress_events:
                    progress_text = progress_events[-1].tensor_proto.string_val[0].decode('utf-8')
                    break
        
        if progress_text:
            f.write("## Current Progress\n")
            f.write(f"```\n{progress_text}\n```\n\n")

        # Metrics summary
        f.write("## Key Metrics (Latest)\n")
        f.write("| Metric | Value |\n")
        f.write("| --- | --- |\n")
        
        important_tags = [
            'Loss/Total', 
            'Loss/Reconstruction_Combined', 
            'Loss/KL', 
            'Training/Learning_Rate',
            'Training/Epoch'
        ]
        
        last_step = 0
        found_any = False
        for tag in important_tags:
            if tag in scalars:
                events = ea.Scalars(tag)
                if events:
                    val = events[-1].value
                    step = events[-1].step
                    last_step = max(last_step, step)
                    f.write(f"| {tag} | {val:.6f} |\n")
                    found_any = True
        
        if not found_any:
            f.write("| No scalar data found | - |\n")
        
        f.write(f"\n**Last Step/Epoch:** {last_step}\n\n")

        # Progress Table (sampled)
        if 'Loss/Total' in scalars:
            f.write("## Training History (Sampled)\n")
            f.write("| Step | Total Loss | Recon Loss | KL Loss | LR |\n")
            f.write("| --- | --- | --- | --- | --- |\n")
            
            total_loss_events = ea.Scalars('Loss/Total')
            recon_events = ea.Scalars('Loss/Reconstruction_Combined') if 'Loss/Reconstruction_Combined' in scalars else None
            kl_events = ea.Scalars('Loss/KL') if 'Loss/KL' in scalars else None
            lr_events = ea.Scalars('Training/Learning_Rate') if 'Training/Learning_Rate' in scalars else None
            
            # Sample at most 25 rows
            n = len(total_loss_events)
            step_size = max(1, n // 25)
            
            indices = list(range(0, n, step_size))
            if (n - 1) not in indices:
                indices.append(n - 1)
            
            for i in indices:
                e = total_loss_events[i]
                step = e.step
                loss = e.value
                
                # Try to find corresponding values at same step
                recon = "-"
                if recon_events:
                    # Look for closest step or exact index
                    if i < len(recon_events) and recon_events[i].step == step:
                        recon = recon_events[i].value
                    else:
                        for re in recon_events:
                            if re.step == step:
                                recon = re.value
                                break
                
                kl = "-"
                if kl_events:
                    if i < len(kl_events) and kl_events[i].step == step:
                        kl = kl_events[i].value
                    else:
                        for ke in kl_events:
                            if ke.step == step:
                                kl = ke.value
                                break

                lr = "-"
                if lr_events:
                    if i < len(lr_events) and lr_events[i].step == step:
                        lr = lr_events[i].value
                    else:
                        for le in lr_events:
                            if le.step == step:
                                lr = le.value
                                break
                
                recon_str = f"{recon:.6f}" if isinstance(recon, float) else recon
                kl_str = f"{kl:.6f}" if isinstance(kl, float) else kl
                lr_str = f"{lr:.2e}" if isinstance(lr, float) else lr
                
                f.write(f"| {step} | {loss:.6f} | {recon_str} | {kl_str} | {lr_str} |\n")

    print(f"Summary written to {output_file}")

if __name__ == "__main__":
    # Ensure we are in the project root
    project_root = r'B:\Coding Projects\++\Neuro\LatentAudio'
    os.chdir(project_root)
    
    latest_dir = get_latest_log_dir()
    summarize_logs(latest_dir)
