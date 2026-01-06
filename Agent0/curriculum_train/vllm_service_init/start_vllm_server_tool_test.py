import requests

def call_hello(name='abc.json', host='0.0.0.0', port=5000, timeout=60):
    url = f'http://{host}:{port}/hello'
    try:
        resp = requests.get(url, params={'name': name}, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {'error': str(e), 'status_code': getattr(resp, 'status_code', None) if 'resp' in locals() else None}

if __name__ == '__main__':
    print(call_hello('/workspace/curriculum/temp_results/temp_2_767699278265_95664.json'))