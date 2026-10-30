# Wall Crack Analyzer

A professional wall crack identification and repair recommendation tool built with pure Python computer vision and real structural engineering data.

## Features

- **Image Analysis**: Uses Sobel edge detection, PCA orientation analysis, and connected component analysis
- **8 Crack Types**: Hairline, Vertical (minor/major), Stair-Step, Horizontal, Diagonal, Map/Crazing, Thermal
- **Real Data**: All repair methods, costs, and identification criteria from actual structural engineering sources
- **No Mock Data**: Every recommendation is based on real industry knowledge
- **Multiple Outputs**: Annotated visualization, detailed text report, JSON export

## Free Hosting Options

### Option 1: Streamlit Cloud (Recommended - Easiest)
1. Create a free account at [share.streamlit.io](https://share.streamlit.io)
2. Create a new GitHub repository
3. Upload these files:
   - `app.py` (the Streamlit app)
   - `requirements.txt`
4. Connect your GitHub repo to Streamlit Cloud
5. Deploy with one click!

**Pros:** Free forever, custom domain support, automatic HTTPS
**Cons:** Public repos only (for free tier), sleeps after inactivity

### Option 2: Hugging Face Spaces
1. Go to [huggingface.co/spaces](https://huggingface.co/spaces)
2. Create a new Space (select Streamlit as SDK)
3. Upload `app.py` and `requirements.txt`
4. Deploy instantly

**Pros:** Free forever, great for ML/AI demos, community features
**Cons:** GPU limited on free tier (not needed for this app)

### Option 3: Render
1. Go to [render.com](https://render.com)
2. Create a new Web Service
3. Connect your GitHub repo
4. Set build command: `pip install -r requirements.txt`
5. Set start command: `streamlit run app.py --server.port $PORT`

**Pros:** Always on (no sleep), supports private repos
**Cons:** Free tier has some limitations

### Option 4: PythonAnywhere
1. Go to [pythonanywhere.com](https://pythonanywhere.com)
2. Upload files via web interface
3. Install dependencies: `pip install -r requirements.txt`
4. Run with custom WSGI configuration

**Pros:** Free tier available, good for learning
**Cons:** Limited bandwidth, no persistent processes on free tier

## Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py

# Or run the CLI version
python wall_crack_analyzer.py --interactive
```

## Local image-learning model

The application can use a locally trained MobileNetV3 image classifier for the
six labelled crack categories. It does not use a paid API or an LLM.

```bash
# Train from Sample_Images and write models/crack_classifier.pt
python train_model.py

# Later, include high-confidence collected upload candidates in a new run
python train_model.py --include-candidates
```

The folder names in `Sample_Images` are the initial labels. During normal use,
the app automatically saves only very high-confidence model predictions to
`data/candidate_uploads`; users are not asked to confirm results. Candidates
are held for a later batch training run and are never added to the active model
immediately. A model is used by the app only after it reaches the built-in
held-out validation threshold; otherwise the existing geometric analyser stays
active as the safe fallback.

## Data Sources

All crack identification data and repair recommendations are based on real structural engineering sources:

- The Basement Guide (foundation crack types)
- JLB Foundation & Waterproofing (severity framework)
- Black Birch Contracting (structural vs non-structural)
- Golden Bay Foundation Builders (width thresholds)
- Dalinghaus Construction (settlement identification)
- AZTA Engineering (forensic crack pattern analysis)
- ARDEX ENDURA (repair methods and materials)
- Mainmark (underpinning costs 2025)
- AquaGuard Injection (epoxy vs polyurethane)
- Stratum Repair (injection method selection)

## Disclaimer

This tool is based on image processing and structural engineering knowledge. It is **NOT** a substitute for professional structural assessment. Always consult a licensed structural engineer for definitive evaluation of wall cracks.

## License

MIT License
