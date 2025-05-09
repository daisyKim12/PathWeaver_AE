import os
import argparse
import numpy as np

def read_u8bin(fname):
    with open(fname, "rb") as f:
        # 처음 8바이트를 읽어서 rows와 cols를 알아냄
        rows, cols = np.frombuffer(f.read(8), dtype=np.uint32)
        # 나머지 데이터를 읽어들임
        data = np.fromfile(f, dtype=np.uint8).reshape((rows, cols))
    return data, rows, cols

def write_fbin(fname, data):
    # fbin 파일에 저장할 때, shape 정보를 먼저 기록한 후 데이터를 기록
    with open(fname, "wb") as f:
        np.asarray(data.shape, dtype=np.uint32).tofile(f)  # 행렬의 크기 기록
        data.tofile(f)  # 데이터 기록

def main():
    parser = argparse.ArgumentParser(description="Convert u8bin file to fbin format")
    parser.add_argument("input", type=str, help="Input u8bin file path")
    parser.add_argument("output", type=str, help="Output fbin file path")

    args = parser.parse_args()

    # u8bin 파일을 읽고 float32로 변환
    data, rows, cols = read_u8bin(args.input)
    print(f"Read u8bin file with shape ({rows}, {cols})")

    # 데이터를 float32로 변환
    data = data.astype(np.float32)

    # 변환된 데이터를 fbin 파일로 저장
    write_fbin(args.output, data)
    print(f"Data saved to fbin file: {args.output}")

if __name__ == "__main__":
    main()
