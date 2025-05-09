import argparse
import requests
import os

def download_file(url, save_dir):
    try:
        # URL에서 파일 이름 추출
        file_name = os.path.basename(url)
        save_path = os.path.join(save_dir, file_name)
        
        # 이미 파일이 존재하는지 확인
        if os.path.exists(save_path):
            print(f"File already exists at {save_path}. Skipping download.")
            return

        # HTTP GET 요청을 보내서 파일을 다운로드
        response = requests.get(url, stream=True)
        response.raise_for_status()  # 요청이 성공했는지 확인

        # 파일을 지정된 경로에 저장
        with open(save_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):  # 파일을 청크 단위로 다운로드
                if chunk:  # 청크가 존재할 때만 기록
                    file.write(chunk)
        
        print(f"File downloaded successfully and saved to {save_path}")
    
    except requests.exceptions.RequestException as e:
        print(f"Error occurred while downloading file: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, required=True, help="The URL of the file to download")
    parser.add_argument("--save_dir", type=str, required=True, help="The directory where the file will be saved")

    args = parser.parse_args()

    # 파일 다운로드 함수 호출
    download_file(args.url, args.save_dir)

if __name__ == "__main__":
    main()
