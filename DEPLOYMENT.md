# Deployment Guide

## Vercel Deployment

This project is configured and ready to be deployed on Vercel. Follow these steps:

### Prerequisites
- Vercel account (https://vercel.com)
- GitHub account with access to this repository

### Deployment Steps

1. **Connect Repository to Vercel**
   - Go to https://vercel.com/new
   - Import this GitHub repository
   - Vercel will auto-detect that this is a Node.js + Vite project

2. **Configure Build Settings**
   - **Root Directory**: `mvp-mock-ui`
   - **Build Command**: `npm run build`
   - **Output Directory**: `dist`
   - **Install Command**: `npm install`

3. **Environment Variables** (if needed)
   - Set `DEEPSEEK_API_KEY` and other environment variables in Vercel project settings
   - Copy values from `.env.local` if using external APIs

4. **Deploy**
   - Click "Deploy"
   - Vercel will handle the rest automatically
   - Your app will be live at `https://your-project-name.vercel.app`

### Important Notes

- The application is configured with standard Vite settings (no custom base paths)
- The thumbnail proxy endpoint (`/thumbnail`) will not work on Vercel unless you implement a serverless function to handle it
- For thumbnail loading, consider:
  - Using Vercel serverless functions (`api/` directory)
  - Using an external image proxy service
  - Loading thumbnails directly from public TikTok URLs if available

### Project Structure

```
mvp-mock-ui/
├── src/                 # React source code
│   ├── components/      # React components
│   ├── services/        # API and utility services
│   └── data/            # Demo data
├── dist/                # Build output (generated)
├── vite.config.ts       # Vite configuration
├── tsconfig.json        # TypeScript configuration
└── package.json         # Dependencies and scripts
```

### Troubleshooting

- **Build fails**: Ensure all dependencies are installed (`npm install`)
- **Images don't load**: Check if the image proxy endpoint needs to be implemented as a serverless function
- **API calls fail**: Verify environment variables are set correctly in Vercel project settings
