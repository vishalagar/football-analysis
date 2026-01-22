import requests
import zipfile
import os
import io

BASE_URL = "http://localhost:8000/api"

def create_dummy_zip():
    # Create a dummy zip file in memory
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        zip_file.writestr('test.txt', 'This is a test file.')
        zip_file.writestr('images/test_img.txt', 'Fake image content')
    
    zip_buffer.seek(0)
    return zip_buffer

def test_upload():
    print("Testing upload...")
    zip_obj = create_dummy_zip()
    files = {'file': ('test_dataset.zip', zip_obj, 'application/zip')}
    
    try:
        response = requests.post(f"{BASE_URL}/upload_dataset", files=files)
        if response.status_code == 200:
            print("Upload Success:", response.json())
        else:
            print("Upload Failed:", response.status_code, response.text)
    except Exception as e:
        print(f"Request failed: {e}")
        print("Make sure the backend server is running on localhost:8000")

if __name__ == "__main__":
    test_upload()
