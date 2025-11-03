# Presentation Guide

## Overview

Your presentation is created using **Slidev** - a markdown-based presentation framework with beautiful themes, animations, and developer-friendly features.

## 📁 Files

- `slides.md` - Main presentation file (Slidev format)
- `package.json` - Node.js dependencies for Slidev
- `report.md` - Full 4-page report with all details

## 🚀 Running the Presentation

### Option 1: Slidev (Interactive, Recommended)

**Install dependencies:**
```bash
npm install
# or
bun install
```

**Run in development mode:**
```bash
npm run dev
# or
bun run dev
```

This will open an interactive presentation at `http://localhost:3030` with:
- Live reload when you edit slides
- Presenter notes
- Drawing tools
- Navigation controls
- Export options

**Export to PDF:**
```bash
npm run export-pdf
```

This generates `slides.pdf` ready for submission.

### Option 2: Convert to PDF (No Slidev Required)

If you don't want to use Slidev, you can convert the report to PDF:

**Using Pandoc:**
```bash
# Install pandoc first: brew install pandoc
pandoc report.md -o report.pdf \
  --pdf-engine=xelatex \
  --variable geometry:margin=1in \
  --variable fontsize=11pt \
  --toc
```

**Using Markdown to PDF tools:**
- **VSCode**: Install "Markdown PDF" extension, right-click → "Markdown PDF: Export (pdf)"
- **Typora**: Open report.md, File → Export → PDF
- **Marked 2** (Mac): Open report.md, File → Export as PDF

### Option 3: Google Slides / PowerPoint

Copy content from `slides.md` to:
1. **Google Slides**: Import markdown or copy section by section
2. **PowerPoint**: Create slides manually using the content
3. **Keynote**: Similar to PowerPoint

## 📊 Slidev Features

### Navigation
- **Arrow keys**: Navigate slides
- **Space**: Next slide
- **Overview mode**: Press `O` to see all slides
- **Presenter mode**: Press `P` (shows notes, timer, next slide)
- **Drawing mode**: Press `D` to draw on slides

### Slides Structure

The presentation has **7 main slides + 3 appendix slides**:

1. **Title Slide**: Introduction and overview
2. **Problem & Approach**: Research question and setup
3. **Quantitative Results**: Metrics and numbers
4. **Qualitative Comparison**: Visual results (images)
5. **Failure Modes & Mitigations**: Technical deep dive
6. **Key Insights**: Trade-offs and comparisons
7. **Conclusions**: Summary and future work
8. **Thank You**: Q&A slide
9. **Appendix**: Training curves, architecture, references

### Customization

Edit `slides.md` to customize:

**Change theme:**
```yaml
---
theme: seriph  # Try: default, apple-basic, shibainu
---
```

**Adjust colors/fonts:**
Add custom CSS at the end of slides.md:
```markdown
<style>
h1 { color: #4CAF50; }
</style>
```

**Add animations:**
Use `<v-click>` for step-by-step reveals (already included)

## 🎨 Using Your Results

All your generated images are automatically referenced:
- `/results/vae_recons.png`
- `/results/vae_interpolation.png`
- `/results/gan_samples.png`
- `/figures/vae_training_history.png`
- `/figures/gan_training_history.png`

The presentation will show these images when you run it!

## 📝 Presentation Tips

### Timing (15 minutes total)
- Slide 1-2 (3 min): Introduction and setup
- Slide 3 (3 min): Quantitative results
- Slide 4 (3 min): Visual comparison
- Slide 5 (4 min): Failure modes (most important!)
- Slide 6-7 (2 min): Conclusions and future work

### What to Emphasize

1. **Trade-offs are fundamental**: Not engineering issues
2. **Failure modes**: Explain WHY they happen, not just HOW to fix
3. **Mitigations are principled**: Based on research, not trial-and-error
4. **Your results**: 37.69% linear probe shows semantic learning

### Potential Questions

Be ready to answer:
- **Why β=1?** Standard ELBO baseline for fair comparison
- **Why these architectures?** Standard, proven, comparable capacity
- **Can VAE match GAN quality?** No with MSE alone, yes with perceptual loss
- **Can GAN be more stable?** Yes with spectral norm + TTUR
- **M1 Mac sufficient?** Yes! Demonstrates accessibility of research

## 📤 Submission Checklist

✅ **slides.pdf** - Export from Slidev or convert manually  
✅ **report.pdf** - Convert report.md to PDF  
✅ **code/** - Your notebook.ipynb and reproduce.sh  
✅ **results/** - All generated images and metrics  

## 🎤 Presenting

### Live Demo Option

If presenting live, you can:
1. Run `npm run dev`
2. Open in fullscreen (F11 or Cmd+Shift+F)
3. Use presenter mode (press `P`)
4. Draw on slides to emphasize points (press `D`)

### Static PDF Option

If submitting PDF only:
1. Run `npm run export-pdf`
2. Submit `slides.pdf`
3. Ensure all images are embedded

## 🛠️ Troubleshooting

### "npm not found"
Install Node.js: https://nodejs.org/ or use `brew install node`

### "Images not showing"
Ensure paths are correct relative to project root:
```
/results/vae_recons.png  ✅ Correct
./results/vae_recons.png ✅ Also works
results/vae_recons.png   ❌ Missing leading slash
```

### "Slidev won't install"
Try using bun instead:
```bash
brew install bun
bun install
bun run dev
```

### "Export PDF hangs"
Slidev uses Playwright. If it hangs:
```bash
npx playwright install chromium
npm run export-pdf
```

## 📚 Resources

- **Slidev Docs**: https://sli.dev
- **Markdown Guide**: https://www.markdownguide.org
- **Presentation Tips**: https://sli.dev/guide/presenter-mode

## 🎯 Quick Start (30 seconds)

```bash
# Install and run
npm install && npm run dev

# Open browser at localhost:3030
# Press 'O' for overview
# Press 'P' for presenter mode
# Edit slides.md to customize
```

That's it! Your presentation is ready to go! 🚀

