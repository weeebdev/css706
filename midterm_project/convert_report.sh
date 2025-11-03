#!/bin/bash

# Script to convert report.md to report.pdf
# Requires pandoc or other markdown to PDF tool

echo "Converting report.md to report.pdf..."

# Check if pandoc is installed
if command -v pandoc &> /dev/null; then
    echo "Using pandoc..."
    pandoc report.md -o report.pdf \
        --pdf-engine=xelatex \
        --variable geometry:margin=1in \
        --variable fontsize=11pt \
        --variable colorlinks=true \
        --toc \
        --number-sections
    
    if [ $? -eq 0 ]; then
        echo "✅ Successfully created report.pdf"
        echo "   Location: $(pwd)/report.pdf"
        open report.pdf 2>/dev/null || echo "   Open it manually to review"
    else
        echo "❌ Pandoc conversion failed"
        exit 1
    fi

# Check if markdown-pdf (npm package) is available
elif command -v markdown-pdf &> /dev/null; then
    echo "Using markdown-pdf..."
    markdown-pdf report.md -o report.pdf
    echo "✅ Successfully created report.pdf"

# Fallback instructions
else
    echo "❌ No PDF conversion tool found"
    echo ""
    echo "Please install pandoc or use one of these methods:"
    echo ""
    echo "1. Install pandoc:"
    echo "   brew install pandoc"
    echo "   Then run this script again"
    echo ""
    echo "2. Use VSCode:"
    echo "   - Install 'Markdown PDF' extension"
    echo "   - Right-click report.md → 'Markdown PDF: Export (pdf)'"
    echo ""
    echo "3. Use online converter:"
    echo "   - Visit: https://www.markdowntopdf.com/"
    echo "   - Upload report.md"
    echo ""
    echo "4. Copy to Google Docs:"
    echo "   - Open report.md in text editor"
    echo "   - Copy all content"
    echo "   - Paste into Google Docs"
    echo "   - File → Download → PDF"
    exit 1
fi

