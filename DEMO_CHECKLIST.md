# OceanEmbed - Hackathon Demo Checklist

## Pre-Demo Setup (Do 1 hour before presentation)

### ✅ Asset Verification
- [ ] Run `python scripts\generate_demo_assets.py` (if not done)
- [ ] Verify all 5 PNG files exist in `assets/` folder
- [ ] Open each PNG to ensure they display correctly
- [ ] Check file sizes (should be 0.3-2.3 MB each)
- [ ] Import all PNGs into your presentation software

### ✅ System Check
- [ ] Close unnecessary applications (free RAM/CPU)
- [ ] Check battery level (plug in if presenting on laptop)
- [ ] Disable system notifications (Windows Focus Assist / Mac Do Not Disturb)
- [ ] Disable screen saver and sleep mode
- [ ] Set screen brightness to maximum
- [ ] Test audio if using voice-over

### ✅ Software Verification
- [ ] Python works: `python --version`
- [ ] Dependencies installed: `pip list | findstr numpy`
- [ ] API can start: `python src\app\api.py` (test for 10 seconds, then Ctrl+C)
- [ ] Dashboard can start: `streamlit run src\app\dashboard.py` (test briefly)
- [ ] Browser launches Streamlit automatically

### ✅ Presentation Materials
- [ ] PowerPoint/Google Slides deck ready
- [ ] All 5 PNG assets imported into slides
- [ ] Slide order matches demo flow
- [ ] Speaker notes prepared
- [ ] Backup PDF export created (in case of software failure)
- [ ] USB drive with all materials (double backup)

### ✅ Demo Environment
- [ ] Test on actual presentation hardware (if possible)
- [ ] Test projector/screen resolution
- [ ] Verify colors look good on projection
- [ ] Check text readability from back of room
- [ ] Audio check (if playing demo video)

---

## 5 Minutes Before Demo

### ✅ Final Checks
- [ ] Close all unnecessary browser tabs
- [ ] Close all unnecessary applications
- [ ] Open only: PowerPoint, 2 terminals (for API/Dashboard)
- [ ] Set browser to fullscreen mode (F11)
- [ ] Mute notifications
- [ ] Charge level > 50% (if on battery)

### ✅ Quick Test
- [ ] Open assets folder: `explorer assets\`
- [ ] Launch demo: `.\launch_demo.ps1 -Both`
- [ ] Wait for dashboard to open (http://localhost:8501)
- [ ] Verify dark theme is applied
- [ ] Close dashboard and API (will restart during actual demo)

### ✅ Mental Preparation
- [ ] Review key metrics (RMSE ~0.9°C, r > 0.95)
- [ ] Review demo flow (Scene 1-8)
- [ ] Practice opening sentence
- [ ] Deep breath - you've got this! 🌊

---

## During Demo - Scene by Scene

### Scene 1: Introduction (30s)
**Visual**: Title slide

**Checklist**:
- [ ] Clear, confident voice
- [ ] State problem clearly
- [ ] Mention INCOIS/MoES

**Key Points**:
- Subsurface temperature critical for forecasting
- In-situ data sparse
- OceanEmbed solves this

### Scene 2: Spatial Capability (45s)
**Visual**: `01_spatial_depth_slices.png`

**Checklist**:
- [ ] Point to Bay of Bengal warm pool
- [ ] Point to Arabian Sea upwelling
- [ ] Emphasize 5 depth levels
- [ ] Note prediction accuracy

**Key Points**:
- Reconstructs temperature at any depth
- Captures regional features
- Ground truth vs prediction comparison

### Scene 3: Vertical Accuracy (45s)
**Visual**: `02_vertical_profiles.png`

**Checklist**:
- [ ] Trace temperature curve
- [ ] Point out thermocline region
- [ ] Mention RMSE value
- [ ] Highlight correlation coefficient

**Key Points**:
- 0-1000m depth reconstruction
- Sub-1.5°C RMSE
- Correlation > 0.95

### Scene 4: Performance Metrics (30s)
**Visual**: `03_error_analysis.png`

**Checklist**:
- [ ] Show spatial RMSE map
- [ ] Point to error histogram
- [ ] Read mean RMSE value
- [ ] Note low errors in open ocean

**Key Points**:
- Basin-wide error analysis
- Mean RMSE ~0.9°C
- Consistent performance

### Scene 5: Interpretable AI (30s)
**Visual**: `04_latent_embeddings.png`

**Checklist**:
- [ ] Explain t-SNE visualization
- [ ] Point to distinct clusters
- [ ] Mention physical meaning
- [ ] Note model learns physics

**Key Points**:
- Model learns representations
- Physically meaningful clusters
- Not just pattern matching

### Scene 6: Comprehensive Evaluation (30s)
**Visual**: `05_depth_wise_metrics.png`

**Checklist**:
- [ ] Show RMSE by depth
- [ ] Note thermocline region
- [ ] Highlight correlation > 0.90
- [ ] Emphasize all-depth performance

**Key Points**:
- 15 depth levels evaluated
- Depth-stratified metrics
- Exceeds targets

### Scene 7: Live Dashboard Demo (60s)
**Visual**: Streamlit Dashboard

**Checklist**:
- [ ] Dashboard already running
- [ ] Browser in fullscreen (F11)
- [ ] Dark theme visible

**Actions**:
1. Surface Inputs Tab (15s)
   - [ ] Show SST map
   - [ ] Switch to SSS
   - [ ] Mention 8 input channels

2. Depth Maps Tab (15s)
   - [ ] Move depth slider 0m → 500m
   - [ ] Show statistics panel
   - [ ] Mention interactivity

3. Depth Profiles Tab (15s)
   - [ ] Click on Bay of Bengal
   - [ ] Show temperature curve
   - [ ] Point to RMSE value

4. 3D Visualization Tab (15s)
   - [ ] Rotate 3D plot
   - [ ] Zoom in/out
   - [ ] Mention operational use

### Scene 8: Impact & Next Steps (30s)
**Visual**: Final slide

**Checklist**:
- [ ] Summarize key benefits
- [ ] Mention applications
- [ ] State next steps
- [ ] Thank judges/audience

**Key Points**:
- Daily basin-wide monitoring
- Cyclone forecasting
- Climate research
- Next: Real data, deployment

---

## Post-Demo

### ✅ Immediate Actions
- [ ] Take screenshot of any errors (for later fixing)
- [ ] Note any questions you couldn't answer
- [ ] Save any feedback received

### ✅ Cleanup
- [ ] Close dashboard (Ctrl+C in terminal)
- [ ] Close API (Ctrl+C in terminal)
- [ ] Re-enable notifications
- [ ] Re-enable screen saver

### ✅ Debrief
- [ ] What went well?
- [ ] What could be improved?
- [ ] Any technical issues?
- [ ] Audience reaction?

---

## Emergency Procedures

### If Dashboard Won't Start
**Backup Plan**:
1. Show static plots instead
2. Narrate as if live
3. Say: "Here's what the interface looks like..."
4. Continue with slides

**Quick Fix** (if time allows):
```powershell
# Kill Streamlit process
taskkill /F /IM streamlit.exe
# Restart
streamlit run src\app\dashboard.py
```

### If API Won't Start
**Impact**: Dashboard may work partially (if using mock data)
**Backup Plan**: Focus on visualizations, mention API exists

**Quick Fix**:
```powershell
# Check port
netstat -ano | findstr :8000
# Kill if occupied
taskkill /PID <pid> /F
# Restart
python src\app\api.py
```

### If Plots Don't Display
**Backup Plan**:
1. Have plots open in image viewer
2. Alt+Tab to show them
3. Or show from assets folder

### If Computer Freezes
**Backup Plan**:
1. Have slides on USB drive
2. Use backup laptop/phone
3. Or continue verbally with printed slides

### If Question You Can't Answer
**Response**:
- "Great question! Let me note that down and get back to you"
- "That's in our roadmap for future work"
- "I'd need to consult my team on the specifics"
- Never make up an answer!

---

## Key Metrics Reference Card

**Print this out and keep handy:**

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OCEANEMBED - QUICK REFERENCE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Domain:
  • North Indian Ocean
  • 5°N to 30°N, 45°E to 105°E
  • 0.25° × 0.25° resolution (~25 km)

Input:
  • 8 satellite surface channels
  • SST, SSS, SSH, currents, winds, mask

Output:
  • 15 depth levels (0-1000m)
  • Full 3D temperature field

Model:
  • ConvNeXt encoder (~28M params)
  • UNet decoder (~14M params)
  • Total: ~42M parameters

Performance:
  • RMSE: 0.6-1.2°C (demo data)
  • Correlation: > 0.95
  • Inference: 0.3s (GPU) / 3s (CPU)
  • Training: 2-4 hours (GPU)

Applications:
  • Cyclone intensity forecasting
  • Marine heatwave detection
  • Climate monitoring
  • Ocean state estimation

Next Steps:
  • Real satellite data integration
  • ARGO validation
  • Operational deployment at INCOIS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## Time Management

**Total Demo Time**: 5 minutes (recommended)

| Scene | Duration | Cumulative |
|-------|----------|-----------|
| Introduction | 30s | 0:30 |
| Spatial Capability | 45s | 1:15 |
| Vertical Accuracy | 45s | 2:00 |
| Performance Metrics | 30s | 2:30 |
| Interpretable AI | 30s | 3:00 |
| Comprehensive Evaluation | 30s | 3:30 |
| Live Dashboard Demo | 60s | 4:30 |
| Impact & Next Steps | 30s | 5:00 |

**Practice with a timer!** Aim to finish in 4:30 to leave buffer.

---

## Success Indicators

✅ Clear, confident delivery  
✅ All 5 plots displayed  
✅ Dashboard runs smoothly  
✅ No technical errors  
✅ Stayed within time limit  
✅ Answered questions well  
✅ Audience engaged  
✅ Judges nodded approvingly  

---

## Final Confidence Boost

**Remember**:
- You've built something impressive
- The visualizations speak for themselves
- If tech fails, you know the content
- Judges want you to succeed
- Passion > perfection
- You've got this! 🌊🚀

---

**Print this checklist and check off items as you go!**

*Last Updated: August 21, 2026*  
*Estimated Prep Time: 30 minutes*  
*Recommended Practice Runs: 2-3 times*
