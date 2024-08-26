# Remove Image Background

This is an open-source project that allows users to remove the background from
images using AI.

## Features

- Upload images via drag and drop or file selection
- Remove background from uploaded images
- Display original and processed images side by side
- Responsive design for desktop and mobile

## Technologies Used

- FastAPI
- Python
- HTML/CSS (Tailwind CSS)
- JavaScript
- Docker

## Installation

1. Clone the repository:

   ```
   git clone https://github.com/drumst0ck/remove-bg.git
   ```

2. Navigate to the project directory:

   ```
   cd remove-bg
   ```

3. Build and run the Docker container:

   ```
   docker build -t remove-bg .
   docker run -p 80:80 remove-bg
   ```

4. Open your browser and visit `http://localhost:80`

## Usage

1. Upload an image by clicking on the upload area or dragging and dropping a
   file.
2. Click the "Remove Background" button.
3. Wait for the image to be processed.
4. View the original and processed images side by side.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [RMBG-1.4 model](https://huggingface.co/briaai/RMBG-1.4)
- [Tailwind CSS](https://tailwindcss.com/)
