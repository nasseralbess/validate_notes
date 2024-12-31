from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
import asyncio

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Placeholder functions for processing
async def process_files(summary_file, detailed_file):
    # Replace with the actual implementation from your script
    summary_content = summary_file.read().decode('utf-8')
    detailed_content = detailed_file.read().decode('utf-8')

    # Mock result
    return {
        "True": [
            {
                "question": "Is the AMIR approach designed to address challenges in AIR?",
                "index": 1,
                "support_text": "The AMIR approach is explicitly designed for AIR challenges."
            }
        ],
        "False": [
            {
                "question": "Does the system utilize a dictionary for root extraction?",
                "index": 3,
                "misalignment_text": "No mention of dictionary usage for root extraction."
            }
        ]
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
async def upload():
    if 'summary' not in request.files or 'detailed' not in request.files:
        return jsonify({"error": "Both files are required."}), 400

    summary_file = request.files['summary']
    detailed_file = request.files['detailed']

    if summary_file.filename == '' or detailed_file.filename == '':
        return jsonify({"error": "Files must have a valid name."}), 400

    summary_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(summary_file.filename))
    detailed_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(detailed_file.filename))

    summary_file.save(summary_path)
    detailed_file.save(detailed_path)

    # Process files
    results = await process_files(summary_file, detailed_file)

    return render_template('results.html', results=results)

if __name__ == '__main__':
    app.run(debug=True)


    
