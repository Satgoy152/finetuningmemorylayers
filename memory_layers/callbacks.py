from transformers import TrainerCallback
import torch
import os
import shutil

class MemoryLayerMonitorAndCheckpoint(TrainerCallback):
    """
    Combined callback for:
    1.  Monitoring memory layer training health
    2. Safe checkpoint saving with safetensors
    """
    
    def __init__(self, model, layers_to_check=[6, 12, 18], 
                 save_every=500, keep_last=2, monitor_every=50):
        # Monitoring
        self.model = model
        self.layers_to_check = layers_to_check
        self.monitor_every = monitor_every
        self.initial_params = {}
        
        # Checkpointing
        self.save_every = save_every
        self.keep_last = keep_last
        self.checkpoints = []
        
        # Store initial parameter values for monitoring
        for idx in layers_to_check:
            layer = model.model.layers[idx].mlp
            self.initial_params[f"layer_{idx}_keys"] = layer.keys.data.clone()
            self.initial_params[f"layer_{idx}_values"] = layer.values.weight.data.clone()
    
    def on_step_end(self, args, state, control, model=None, tokenizer=None, **kwargs):
        step = state.global_step
        
        # ================================================================
        # MONITORING (every N steps)
        # ================================================================
        if step % self.monitor_every == 0 and step > 0:
            self._monitor_health(step)
        
        # ================================================================
        # CHECKPOINTING (every M steps)
        # ================================================================
        if step % self.save_every == 0 and step > 0:
            self._save_checkpoint(step, state, model, tokenizer)
    
    def _monitor_health(self, step):
        """Monitor memory layer training health"""
        print(f"\n{'='*80}")
        print(f"🔍 MEMORY LAYER HEALTH CHECK - Step {step}")
        print(f"{'='*80}")
        
        all_healthy = True
        
        for idx in self.layers_to_check:
            layer = self.model.model.layers[idx].mlp
            
            # Check parameter changes
            keys_diff = (
                layer.keys.data - self.initial_params[f"layer_{idx}_keys"]
            ).abs().mean().item()
            values_diff = (
                layer.values.weight.data - self.initial_params[f"layer_{idx}_values"]
            ).abs().mean().item()
            
            # Check gradients
            keys_grad = layer.keys.grad.norm().item() if layer.keys.grad is not None else 0.0
            values_grad = (
                layer.values.weight.grad.norm().item() 
                if layer.values.weight.grad is not None else 0.0
            )
            
            # Parameter statistics
            keys_mean = layer.keys.data.mean().item()
            keys_std = layer.keys.data.std().item()
            values_mean = layer.values.weight.data.mean().item()
            values_std = layer.values.weight.data.std().item()
            
            print(f"\n📊 Layer {idx} Memory:")
            print(f"  Parameters:")
            print(f"    Keys:   mean={keys_mean:+.4f}, std={keys_std:.4f}")
            print(f"    Values: mean={values_mean:+.4f}, std={values_std:.4f}")
            print(f"  Changes since start:")
            print(f"    Keys:   {keys_diff:.6f} {'✅' if keys_diff > 1e-6 else '❌ FROZEN'}")
            print(f"    Values: {values_diff:.6f} {'✅' if values_diff > 1e-6 else '❌ FROZEN'}")
            print(f"  Gradient norms:")
            print(f"    Keys:   {keys_grad:.4f} {'✅' if keys_grad > 0 else '❌ NO GRAD'}")
            print(f"    Values: {values_grad:.4f} {'✅' if values_grad > 0 else '❌ NO GRAD'}")
            
            # Health checks
            if keys_diff < 1e-8 and step > 100:
                print(f"  ⚠️  WARNING: Keys not updating!")
                all_healthy = False
            if values_diff < 1e-8 and step > 100:
                print(f"  ⚠️  WARNING: Values not updating!")
                all_healthy = False
            if keys_grad == 0.0:
                print(f"  ⚠️  WARNING: No gradient flow to keys!")
                all_healthy = False
            if values_grad == 0.0:
                print(f"  ⚠️  WARNING: No gradient flow to values!")
                all_healthy = False
        
        if all_healthy:
            print(f"\n✅ All memory layers healthy!")
        else:
            print(f"\n⚠️  Some memory layers need attention!")
        
        print(f"{'='*80}\n")
    
    def _save_checkpoint(self, step, state, model, tokenizer):
        """Save checkpoint safely with safetensors"""
        checkpoint_dir = f"./checkpoints/step-{step}"
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        print(f"\n💾 Saving checkpoint at step {step}...")
        
        try:
            # Save model with safetensors (no JSON serialization issues)
            model.save_pretrained(
                checkpoint_dir, 
                safe_serialization=True
            )
            
            # Save tokenizer
            if tokenizer:
                tokenizer.save_pretrained(checkpoint_dir)
            
            # Save minimal training state (safe to serialize)
            training_state = {
                'step': step,
                'epoch': state.epoch,
                'global_step': state.global_step,
            }
            
            # Add last loss if available
            if state.log_history:
                last_log = state.log_history[-1]
                if 'loss' in last_log:
                    training_state['loss'] = last_log['loss']
            
            torch.save(
                training_state, 
                os.path.join(checkpoint_dir, 'training_state.pt')
            )
            
            # Track checkpoints
            self.checkpoints.append(checkpoint_dir)
            
            # Remove old checkpoints (keep only last N)
            if len(self.checkpoints) > self.keep_last:
                old_checkpoint = self.checkpoints.pop(0)
                if os.path.exists(old_checkpoint):
                    shutil.rmtree(old_checkpoint)
                    print(f"  🗑️  Removed old checkpoint: {os.path.basename(old_checkpoint)}")
            
            print(f"  ✅ Checkpoint saved: {checkpoint_dir}")
            
        except Exception as e:
            print(f"  ❌ Failed to save checkpoint: {e}")
            # Continue training even if checkpoint fails
    
    def on_train_end(self, args, state, control, model=None, tokenizer=None, **kwargs):
        """Save final model at end of training"""
        print(f"\n{'='*80}")
        print("🏁 TRAINING COMPLETE - Saving final model")
        print(f"{'='*80}\n")
        
        final_dir = "./qwen_memory_final"
        os.makedirs(final_dir, exist_ok=True)
        
        model.save_pretrained(final_dir, safe_serialization=True)
        if tokenizer:
            tokenizer.save_pretrained(final_dir)
        
        # Save final statistics
        final_stats = {
            'total_steps': state.global_step,
            'total_epochs': state.epoch,
        }
        
        if state.log_history:
            losses = [log['loss'] for log in state.log_history if 'loss' in log]
            if losses:
                final_stats['final_loss'] = losses[-1]
                final_stats['initial_loss'] = losses[0]
                final_stats['loss_improvement'] = losses[0] - losses[-1]
        
        torch.save(final_stats, os.path.join(final_dir, 'final_stats.pt'))
        
        print(f"✅ Final model saved to: {final_dir}")
        print(f"   Total steps: {state.global_step}")
        print(f"   Total epochs: {state.epoch:.2f}")
        if 'loss_improvement' in final_stats:
            print(f"   Loss improvement: {final_stats['loss_improvement']:.4f}")
        print(f"\n{'='*80}\n")
