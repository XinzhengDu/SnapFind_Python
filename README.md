# SnapFind
Find any moment in video, with one sentence.

## System Compatibility
This software is **only available for Mac or Linux**.  
For Windows users, please use the dedicated version:  
https://github.com/XinzhengDu/SnapFind  

## Open Source Foundation
This software is built on siglip2 and adheres to open-source principles.

## 100% Local Execution (Privacy Compliance)
Core Logic: All video parsing, model inference, and content retrieval are processed **entirely on the user's device**.  
No user data (videos, images, text) is uploaded to any server. This complies with global privacy regulations:
- China: Personal Information Protection Law (PIPL)
- EU: General Data Protection Regulation (GDPR)
- US: California Consumer Privacy Act (CCPA)

## Important Note
Currently, **English query terms are better supported** than other languages. If search results are inaccurate, translate your query into English before input.

## Installation & Usage
### 1. Download Required Files
First, download the following model and tokenizer files:
- siglip_vision.onnx
- siglip_text.onnx
- tokenizer.json  

Download link: https://www.modelscope.cn/models/XinzhengDu/siglip2-base-patch16-224-onnx/files  
Place all files in the `model` folder of the project.

### 2. Set Up Environment & Install Dependencies
Create a new Python environment and run the following commands to install dependencies:
```bash
pip install torch==2.1.0+cu121 torchvision==0.16.0+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install accelerate==0.30.1 altgraph==0.17.5 autocommand==2.2.2 backports.tarfile==1.0.0 certifi==2022.12.7 charset-normalizer==2.1.1 colorama==0.4.6 docx2txt==0.9 filelock==3.20.0 fsspec==2025.12.0 huggingface-hub==0.36.0 idna==3.4 jaraco.functools==4.4.0 Jinja2==3.1.3 lxml==6.0.2 MarkupSafe==2.1.5 more-itertools==10.8.0 mpmath==1.3.0 networkx==3.6.1 numpy==1.26.4 opencv-python==4.8.0.76 packaging==25.0 pefile==2024.8.26 pillow==12.0.0 psutil==7.2.1 pyinstaller==5.13.2 "pyinstaller-hooks-contrib==2025.11" PyPDF2==3.0.1 PyQt5==5.15.7 "PyQt5-Qt5==5.15.2" PyQt5-sip==12.11.0 python-docx==1.2.0 pywin32-ctypes==0.2.3 PyYAML==6.0.3 regex==2026.1.15 requests==2.28.1 safetensors==0.7.0 scipy==1.17.0 setuptools==65.5.0 sympy==1.14.0 tokenizers==0.21.4 tqdm==4.67.1 transformers==4.52.0 typing_extensions==4.15.0 urllib3==1.26.13
```

## Beta Features (Currently Available)
1. Content matching: Match search terms with videos, images, and documents. Queries support descriptions of time, location, people, and scenarios.  
2. OCR matching: Recognize and match text in videos/images (requires more computing power).  
3. Feature caching: Cache file features to reduce waiting time for subsequent calls.  
4. Search results: View thumbnails and open source files directly from the results page.  
5. Frame interval customization: Set N-second frame capture intervals for long/short videos to extract features.  
6. Precision adjustment: Optimize cache precision for different scenarios to maximize computing resource utilization (Settings > File).  
7. Low-spec device support: Set longer frame intervals for devices with limited computing power (Settings > File).  
8. Top-N results: Return the N most relevant results (more results = longer wait time; reduce N for time-sensitive use cases).

## Upcoming Features (In Development)
- GPU acceleration: 90% complete (supports video loading and model inference via GPU).  
- Image upload for search: 90% complete (search for relevant content by uploading images).  
- Video player integration: Jump to specific timestamps, add screenshot/GIF features (for meme creation), and audio positioning (for video segmenting).  
- Direct file navigation: Open files at specific timestamps (requires designated video player installation).  
- Idle-time feature computation: Automatically compute file features when the computer is idle (pauses if resource-intensive programs run).  
- Real-time monitoring analysis: Analyze risk levels of live surveillance content (e.g., monitor pet status).  
- Vertical model training: Train domain-specific models (requires significant computing power and datasets).

## Use Cases
- Locate video timestamps by human actions (e.g., "find the 10th minute where someone waves hands").  
- Fast navigation in tutorial videos: Jump directly to relevant sections without watching the entire video.  
- Class video indexing: Quickly find key moments in recorded lectures.  
- Surveillance footage analysis: Avoid manual frame-by-frame checking (e.g., "find the moment a package is delivered").  
- Image-based search: Upload a photo (e.g., a group photo) to find related historical images/videos.

## User Agreement & Disclaimer
1. **Legitimate Use Only**: This software is strictly for legal and compliant use cases. It is prohibited for illegal surveillance, invasion of privacy, or other unlawful activities.  
2. **User Liability**: Users assume full responsibility for all consequences of using this software. The developer shall not be liable for any misuse.  
3. **Content Restrictions**: The software restricts retrieval of illegal content (e.g., pornography, violence) to comply with applicable laws and regulations.  
