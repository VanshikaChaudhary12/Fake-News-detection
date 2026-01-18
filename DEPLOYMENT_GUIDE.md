# Deployment Guide

## ⚠️ Important: Vercel Limitations

Vercel has the following limitations that may affect this project:
- **Function size limit**: 50MB (your model files might exceed this)
- **Execution timeout**: 10 seconds for Hobby plan
- **Memory limit**: 1024MB for Hobby plan

## 🚀 Recommended Deployment Options

### Option 1: Render (Recommended for ML Apps)
**Best for**: ML applications with large model files

1. Go to https://render.com
2. Sign up with GitHub
3. Click "New +" → "Web Service"
4. Connect your repository
5. Configure:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`
6. Add to requirements.txt: `gunicorn==21.2.0`
7. Deploy!

**Pros**: 
- Free tier available
- Supports large files
- Better for ML models
- No timeout issues

### Option 2: Railway
**Best for**: Quick deployment

1. Go to https://railway.app
2. Sign in with GitHub
3. Click "New Project" → "Deploy from GitHub repo"
4. Select your repository
5. Railway auto-detects Flask and deploys

**Pros**:
- Very easy setup
- Free $5 credit monthly
- Good for ML apps

### Option 3: PythonAnywhere
**Best for**: Python-specific hosting

1. Go to https://www.pythonanywhere.com
2. Create free account
3. Upload your code
4. Configure WSGI file
5. Deploy

**Pros**:
- Python-focused
- Free tier available
- Good documentation

### Option 4: Heroku
**Best for**: Professional deployment

1. Install Heroku CLI
2. Create `Procfile`: `web: gunicorn app:app`
3. Run:
   ```bash
   heroku login
   heroku create your-app-name
   git push heroku main
   ```

**Pros**:
- Industry standard
- Easy scaling
- Good documentation

## 📝 Files Needed for Deployment

### For Render/Railway/Heroku:
Add to requirements.txt:
```
gunicorn==21.2.0
```

Create `Procfile`:
```
web: gunicorn app:app
```

### For All Platforms:
Ensure you have:
- ✅ requirements.txt
- ✅ app.py
- ✅ model files (model.pkl, vectorizer.pkl)
- ✅ All templates and static files

## 🔧 Quick Fix for Vercel

If you still want to try Vercel, reduce model size:
1. Use a simpler model
2. Compress pickle files
3. Use external storage (AWS S3) for models

## 💡 Recommendation

**Use Render** - It's free, supports ML models, and works great with Flask applications!

Visit: https://render.com
