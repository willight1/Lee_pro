"""
OpenAI API 연결 테스트 스크립트
API 호출 오류 원인을 진단합니다.
"""

import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

def test_api_connection():
    """OpenAI API 연결을 테스트합니다"""
    
    print("=" * 60)
    print("OpenAI API 연결 테스트")
    print("=" * 60)
    
    # 1. API 키 확인
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ 오류: OPENAI_API_KEY가 .env 파일에 설정되지 않았습니다.")
        return False
    
    print(f"✓ API 키 발견: {api_key[:20]}...")
    
    # 2. API 키 형식 확인
    if not api_key.startswith(("sk-", "sk-proj-")):
        print(f"⚠️  경고: API 키 형식이 예상과 다릅니다.")
    else:
        print("✓ API 키 형식 확인됨")
    
    # 3. 클라이언트 초기화 테스트
    try:
        client = OpenAI(api_key=api_key)
        print("✓ OpenAI 클라이언트 초기화 성공")
    except Exception as e:
        print(f"❌ 클라이언트 초기화 실패: {e}")
        return False
    
    # 4. 간단한 API 호출 테스트
    try:
        print("\nAPI 호출 테스트 중...")
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": "Hello, this is a test. Please respond with 'OK'."}
            ],
            max_tokens=10,
            timeout=30
        )
        
        result = response.choices[0].message.content
        print(f"✓ API 호출 성공!")
        print(f"  응답: {result}")
        return True
        
    except Exception as e:
        error_type = type(e).__name__
        error_msg = str(e)
        
        print(f"\n❌ API 호출 실패!")
        print(f"  오류 타입: {error_type}")
        print(f"  오류 메시지: {error_msg}")
        
        # 구체적인 오류 분석
        if "authentication" in error_msg.lower() or "api key" in error_msg.lower() or "invalid" in error_msg.lower():
            print("\n💡 해결 방법:")
            print("  1. .env 파일의 OPENAI_API_KEY가 올바른지 확인하세요")
            print("  2. API 키가 만료되지 않았는지 확인하세요")
            print("  3. OpenAI Platform에서 새로운 API 키를 발급받으세요")
        elif "rate limit" in error_msg.lower() or "quota" in error_msg.lower():
            print("\n💡 해결 방법:")
            print("  1. API 할당량이 초과되었습니다")
            print("  2. 잠시 후 다시 시도하세요")
            print("  3. OpenAI Platform에서 사용량을 확인하세요")
        elif "timeout" in error_msg.lower() or "connection" in error_msg.lower():
            print("\n💡 해결 방법:")
            print("  1. 인터넷 연결을 확인하세요")
            print("  2. 방화벽이나 프록시 설정을 확인하세요")
            print("  3. 네트워크 연결이 안정적인지 확인하세요")
        elif "context_length" in error_msg.lower() or "token" in error_msg.lower():
            print("\n💡 해결 방법:")
            print("  1. 입력 텍스트가 너무 깁니다")
            print("  2. PDF 파일을 더 작은 청크로 나누어 처리하세요")
        
        return False

if __name__ == "__main__":
    success = test_api_connection()
    
    print("\n" + "=" * 60)
    if success:
        print("✅ 모든 테스트 통과!")
    else:
        print("❌ 테스트 실패 - 위의 오류 메시지를 확인하세요")
    print("=" * 60)

