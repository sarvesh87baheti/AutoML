from flask import Flask, render_template, request, jsonify
import os
from pathlib import Path
from runner import run_pipeline

app = Flask(__name__)

# Folder to store uploaded datasets
UPLOAD_FOLDER = "uploaded_files"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_file():
    try:
        # ----------- 1. Read uploaded file ------------
        file = request.files.get("dataset")
        if not file:
            return jsonify({"error": "No file uploaded"}), 400

        filename = file.filename
        save_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(save_path)

        # ----------- 2. Read form inputs ------------
        problem_type = request.form.get("problem_type")
        target_col = request.form.get("target_col", None)

        if problem_type in ["regression", "classification"] and not target_col:
            return jsonify({"error": "Target column required for supervised tasks"}), 400

        # ----------- 3. Run AutoML Pipeline ------------
        results = run_pipeline(
            file_path=save_path,
            problem_type=problem_type,
            target_col=target_col
        )

        return jsonify({
            "message": "AutoML process completed",
            "results": results
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
