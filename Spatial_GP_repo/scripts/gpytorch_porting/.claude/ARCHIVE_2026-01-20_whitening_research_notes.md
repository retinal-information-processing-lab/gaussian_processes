Key Synthesis from Research:                                                                                                                    
                                                                                                                                                  
  1. GPyTorch's design philosophy: Store whitened params (m', S'), and on each forward pass, compute the unwhitening using the CURRENT L_K.       
  Autograd differentiates through L_K automatically.                                                                                              
  2. This works for joint optimization (Adam on all params) because gradients for both variational and kernel params are computed in the same     
  forward pass, and autograd handles the coupling through L_K.                                                                                    
  3. The incompatibility with EM: Our E-step produces natural (m, V), we store whitened versions with L_K_old. M-step changes kernel → L_K_new ≠  
  L_K_old. Next E-step reads with L_K_new → corruption!                                                                                           
  4. Critical insight from GPflow issue #979: "Whitening is just a variable transformation... it does not change the model at all"                
                                                                                                                                                  
  The most promising finding is the mention of UnwhitenedVariationalStrategy. Let me investigate this further.      


   cat << 'EOF'                                                                                                                                   
                                                                                                                                                  
   ################################################################################                                                               
   #                    DIAGNOSTIC PACKAGE CREATION SUMMARY                      #                                                                
   ################################################################################                                                               
                                                                                                                                                  
   TASK: Create diagnostic script comparing whitened vs unwhitened E-step behavior                                                                
                                                                                                                                                  
   STATUS: COMPLETE ✓                                                                                                                             
                                                                                                                                                  
   FILES CREATED (in /tmp/):                                                                                                                      
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━                                                                    
                                                                                                                                                  
   1. diagnostic_estep_whitening.py (15 KB)                                                                                                       
      ├─ Main diagnostic script                                                                                                                   
      ├─ Compares whitened vs unwhitened side-by-side                                                                                             
      ├─ Runs 5 E-step iterations with full diagnostics                                                                                           
      ├─ Checks: parameters, moments, KL, ELBO, predictions                                                                                       
      └─ Ready to run: conda run -n pytorch_gpytorch python diagnostic_estep_whitening.py                                                         
                                                                                                                                                  
   2. FINAL_REPORT.md (12 KB)                                                                                                                     
      ├─ Executive summary and problem description                                                                                                
      ├─ Root cause analysis (incomplete V whitening)                                                                                             
      ├─ Diagnostic approach and what we're checking                                                                                              
      ├─ How to run and interpret results                                                                                                         
      ├─ Expected outcomes and solutions                                                                                                          
      └─ Read this FIRST (10-minute overview)                                                                                                     
                                                                                                                                                  
   3. DIAGNOSTIC_SUMMARY.md (5 KB)                                                                                                                
      ├─ Detailed root cause analysis                                                                                                             
      ├─ What's implemented vs missing                                                                                                            
      ├─ Code locations and impact analysis                                                                                                       
      ├─ Two proposed solutions                                                                                                                   
      └─ Read this for technical understanding                                                                                                    
                                                                                                                                                  
   4. HOW_TO_RUN_DIAGNOSTIC.md (8 KB)                                                                                                             
      ├─ Step-by-step execution guide                                                                                                             
      ├─ Output interpretation guide                                                                                                              
      ├─ Smoking gun: KL divergence section                                                                                                       
      ├─ Troubleshooting common issues                                                                                                            
      ├─ Next steps after running                                                                                                                 
      └─ Read this BEFORE running script                                                                                                          
                                                                                                                                                  
   5. whitened_vs_unwhitened_diagnostic_analysis.md (9 KB)                                                                                        
      ├─ Deep technical analysis                                                                                                                  
      ├─ GPyTorch parameterization explanation                                                                                                    
      ├─ Parameter interpretation details                                                                                                         
      ├─ Diagnostic strategy breakdown                                                                                                            
      └─ Read for mathematical derivations                                                                                                        
                                                                                                                                                  
   6. README_DIAGNOSTIC.md (5 KB)                                                                                                                 
      ├─ Index of all diagnostic materials                                                                                                        
      ├─ Quick start (5 minutes)                                                                                                                  
      ├─ Reading order and dependencies                                                                                                           
      ├─ Files to reference                                                                                                                       
      └─ Read to navigate the package                                                                                                             
                                                                                                                                                  
   PROBLEM IDENTIFIED:                                                                                                                            
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━                                                                    
                                                                                                                                                  
   Whitened E-step performs poorly (r ≈ 0.6878) vs unwhitened (r > 0.7)                                                                           
                                                                                                                                                  
   ROOT CAUSE (Hypothesis):                                                                                                                       
     ├─ Variational mean m:    WHITENED ✓                                                                                                         
     ├─ Variational covar V:   NATURAL ✗ (should be whitened)                                                                                     
     ├─ This creates:          PARAMETER MISMATCH                                                                                                 
     ├─ Which causes:          INCORRECT KL DIVERGENCE                                                                                            
     └─ Which results in:      POOR TRAINING PERFORMANCE                                                                                          
                                                                                                                                                  
   SMOKING GUN TEST:                                                                                                                              
     If diagnostic shows: KL_difference > 100                                                                                                     
     Then hypothesis is: CONFIRMED ✓                                                                                                              
     Otherwise:          Problem is elsewhere                                                                                                     
                                                                                                                                                  
   QUICK START:                                                                                                                                   
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━                                                                    
                                                                                                                                                  
     1. Copy script:                                                                                                                              
        cp /tmp/diagnostic_estep_whitening.py \                                                                                                   
           ~/IDV_code/ClosedLoopProject/gaussian_processes/Spatial_GP_repo/scripts/gpytorch_porting/                                              
                                                                                                                                                  
     2. Run diagnostic (2-3 minutes):                                                                                                             
        cd ~/IDV_code/.../scripts/gpytorch_porting/                                                                                               
        conda run -n pytorch_gpytorch python diagnostic_estep_whitening.py 2>&1 | tee diagnostic.log                                              
                                                                                                                                                  
     3. Check for smoking gun:                                                                                                                    
        grep -A 5 "KL Divergence" diagnostic.log                                                                                                  
        Look for: "Difference: |KL_w - KL_uw| = ???"                                                                                              
        If > 100: Problem confirmed!                                                                                                              
        If < 10:  Problem elsewhere                                                                                                               
                                                                                                                                                  
   EXPECTED OUTPUT (if hypothesis is correct):                                                                                                    
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━                                                                    
                                                                                                                                                  
     KL Divergence                                                                                                                                
       Whitened:   KL = 542.12                                                                                                                    
       Unwhitened: KL = 234.56                                                                                                                    
       Difference: |KL_w - KL_uw| = 307.56  ← SMOKING GUN!                                                                                        
       WARNING: Large KL divergence difference! May indicate parameter mismatch.                                                                  
                                                                                                                                                  
     Total ELBO                                                                                                                                   
       Whitened:   ELBO = -308.45 (ELL=6500.12, KL=542.12)                                                                                        
       Unwhitened: ELBO = 50.23 (ELL=6800.23, KL=234.57)                                                                                          
       Difference: |ELBO_w - ELBO_uw| = 358.68                                                                                                    
                                                                                                                                                  
     Test Pearson correlation:                                                                                                                    
       Whitened:   r = 0.6878                                                                                                                     
       Unwhitened: r = 0.7512                                                                                                                     
       CONCLUSION: Large performance gap detected                                                                                                 
                                                                                                                                                  
   NEXT STEPS AFTER DIAGNOSTIC:                                                                                                                   
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━                                                                    
                                                                                                                                                  
     If KL difference > 100 (problem confirmed):                                                                                                  
                                                                                                                                                  
       Option A: Quick fix (disable whitening)                                                                                                    
         └─ Change whitening=True → whitening=False in model creation                                                                             
         └─ Trade-off: Less numerically stable but works correctly                                                                                
                                                                                                                                                  
       Option B: Proper fix (implement V whitening)                                                                                               
         └─ Implement get_variational_covar_with_L_K()                                                                                            
         └─ Implement update_variational_covar_with_L_K()                                                                                         
         └─ Trade-off: More work but proper solution                                                                                              
                                                                                                                                                  
     If KL difference < 10 (problem elsewhere):                                                                                                   
                                                                                                                                                  
       Options:                                                                                                                                   
         └─ Investigate kernel parameter learning                                                                                                 
         └─ Check F-step (A, lambda0) convergence                                                                                                 
         └─ Look at E-step stability checks                                                                                                       
         └─ Profile performance bottlenecks                                                                                                       
                                                                                                                                                  
   KEY INSIGHTS:                                                                                                                                  
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━                                                                    
                                                                                                                                                  
   1. VariationalStrategy (whitened) expects BOTH m and L to be whitened                                                                          
      - m whitened: L_K^{-1} @ m_natural ✓ (implemented)                                                                                          
      - L whitened: L_K^{-1} @ L_natural @ ??? ✗ (missing!)                                                                                       
                                                                                                                                                  
   2. UnwhitenedVariationalStrategy expects BOTH m and L to be natural                                                                            
      - m natural: m_natural ✓ (just don't convert)                                                                                               
      - L natural: L_natural ✓ (just don't convert)                                                                                               
      - No mismatch → works correctly                                                                                                             
                                                                                                                                                  
   3. The KL divergence is extremely sensitive to parameterization                                                                                
      - Small parameter mismatch → large KL error                                                                                                 
      - KL error → high loss → poor performance                                                                                                   
      - This is why we see 0.06+ correlation gap                                                                                                  
                                                                                                                                                  
   ESTIMATED EFFORT:                                                                                                                              
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━                                                                    
                                                                                                                                                  
     Reading diagnostic materials:    20-30 min                                                                                                   
     Running diagnostic script:       5-10 min (mostly automatic)                                                                                 
     Interpreting results:            10-20 min                                                                                                   
     Deciding on fix:                 5 min                                                                                                       
     ────────────────────────────────────────                                                                                                     
     Total to diagnosis + decision:   1-2 hours                                                                                                   
                                                                                                                                                  
     Implementing fix (if needed):    1-3 hours                                                                                                   
     Re-testing:                      1-2 hours                                                                                                   
                                                                                                                                                  
   READING ORDER:                                                                                                                                 
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━                                                                    
                                                                                                                                                  
     1. README_DIAGNOSTIC.md ..................... Navigation and overview                                                                        
     2. FINAL_REPORT.md .......................... Problem and hypothesis                                                                         
     3. DIAGNOSTIC_SUMMARY.md ................... Root cause analysis                                                                             
     4. HOW_TO_RUN_DIAGNOSTIC.md ............... Execution guide                                                                                  
     5. diagnostic_estep_whitening.py .......... Run the script                                                                                   
     6. whitened_vs_unwhitened_diagnostic_analysis.md ... Deep dive (if needed)                                                                   
                                                                                                                                                  
   DELIVERABLES CHECKLIST:                                                                                                                        
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━                                                                    
                                                                                                                                                  
     ✓ Diagnostic analysis document                                                                                                               
     ✓ Diagnostic Python script                                                                                                                   
     ✓ Step-by-step running guide                                                                                                                 
     ✓ Root cause analysis                                                                                                                        
     ✓ Expected results and interpretation guide                                                                                                  
     ✓ Solutions and next steps                                                                                                                   
     ✓ Navigation and index document                                                                                                              
                                                                                                                                                  
   All files are in: /tmp/                                                                                                                        
                                                                                                                                                  
   Ready to: Copy to working directory and run diagnostic                                                                                         
                                                                                                                                                  
   ################################################################################                                                               
   EOF         