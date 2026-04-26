"""File management routes for Thoth backend."""
from flask import Blueprint, jsonify, request, send_file, abort
from pathlib import Path
import os
import base64
import requests
import logging
from ..file_manager import file_manager
from ..config import Config

logger = logging.getLogger(__name__)

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


@bp.route('/upload-to-cloud/<path:filename>', methods=['POST'])
def upload_to_cloud(filename):
    """Upload a local file to the Brain cloud server.
    
    This endpoint is called by the Research Portal to trigger Thoth to push
    a file to Brain (since Brain cannot reach Thoth due to NAT).
    """
    try:
        # Get auth token from config (set during login)
        auth_token = getattr(Config, 'USER_AUTH_TOKEN', None)
        if not auth_token:
            return jsonify({
                'status': 'error',
                'message': 'Device not authenticated. Please log in first.'
            }), 401
        
        # Get the file path
        file_path = file_manager.get_file(filename)
        if not file_path or not file_path.exists():
            return jsonify({
                'status': 'error',
                'message': f'File not found: {filename}'
            }), 404
        
        # Read file content
        with open(file_path, 'rb') as f:
            content = f.read()
        
        # Encode as base64
        content_b64 = base64.b64encode(content).decode('utf-8')
        
        # Determine content type
        ext = file_path.suffix.lower()
        content_type = 'application/json' if ext == '.json' else 'text/csv' if ext == '.csv' else 'application/octet-stream'
        
        # Get device ID
        device_id = getattr(Config, 'DEVICE_ID', None)
        
        # Upload to Brain
        brain_url = f"{Config.BRAIN_SERVER_URL}/file/upload"
        headers = {
            'Authorization': f'Bearer {auth_token}',
            'Content-Type': 'application/json'
        }
        
        payload = {
            'filename': filename,
            'content': content_b64,
            'is_base64': True,
            'device_id': device_id,
            'content_type': content_type
        }
        
        logger.info(f"Uploading {filename} ({len(content)} bytes) to Brain cloud")
        
        response = requests.post(brain_url, json=payload, headers=headers, timeout=120)
        
        if response.status_code in (200, 201):
            result = response.json()
            logger.info(f"File uploaded successfully: {result}")
            return jsonify({
                'status': 'success',
                'message': 'File uploaded to cloud',
                'cloud_file_id': result.get('file_id'),
                'filename': filename
            })
        else:
            logger.error(f"Upload failed: {response.status_code} - {response.text}")
            return jsonify({
                'status': 'error',
                'message': f'Upload failed: {response.text}'
            }), response.status_code
            
    except requests.exceptions.Timeout:
        return jsonify({
            'status': 'error',
            'message': 'Upload timed out'
        }), 504
    except Exception as e:
        logger.error(f"Error uploading to cloud: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500
