"""File management routes for Thoth backend."""
from flask import Blueprint, jsonify, request, send_file, abort
from pathlib import Path
import os
from ..file_manager import file_manager

bp = Blueprint('files', __name__, url_prefix='/api/files')

@bp.route('', methods=['GET'])
def list_files():
    """List all files in the data directory."""
    try:
        subdir = request.args.get('subdir', '')
        files = file_manager.list_files(subdir)
        return jsonify({
            'status': 'success',
            'data': files,
            'current_dir': subdir
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@bp.route('/download/<path:file_path>', methods=['GET'])
def download_file(file_path):
    """Download a file from the data directory."""
    try:
        file_path = file_manager.get_file(file_path)
        if not file_path:
            abort(404, description="File not found or access denied")
        
        # Get file info for headers
        stat = file_path.stat()
        file_size = stat.st_size
        mtime = stat.st_mtime
        
        # Create response with file
        response = send_file(
            file_path,
            as_attachment=True,
            download_name=file_path.name,
            mimetype='application/octet-stream'
        )
        
        # Set headers
        response.headers['Content-Length'] = file_size
        response.headers['Last-Modified'] = mtime
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '-1'
        
        return response
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@bp.route('/upload', methods=['POST'])
def upload_file():
    """Upload a file to the data directory."""
    try:
        if 'file' not in request.files:
            return jsonify({
                'status': 'error',
                'message': 'No file part in the request'
            }), 400
            
        file = request.files['file']
        subdir = request.form.get('subdir', '')
        
        if file.filename == '':
            return jsonify({
                'status': 'error',
                'message': 'No selected file'
            }), 400
            
        # Create target directory if it doesn't exist
        target_dir = Path(file_manager.base_path) / subdir
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # Save the file
        file_path = target_dir / file.filename
        file.save(file_path)
        
        return jsonify({
            'status': 'success',
            'message': 'File uploaded successfully',
            'path': str(file_path.relative_to(file_manager.base_path))
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500
