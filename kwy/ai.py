import os
import base64
import json
from openai import OpenAI
from pathlib import Path
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()



class KoreanImageAnalyzer:
    """한국 관광지 및 음식 이미지 분석 AI"""
    
    def __init__(self, api_key=None):
        """
        OpenAI API 초기화
        
        Args:
            api_key: OpenAI API 키. None인 경우 환경 변수에서 가져옴
        """
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY가 필요합니다. 환경 변수로 설정하거나 매개변수로 전달하세요.")
        
        self.client = OpenAI(api_key=self.api_key)
    
    def encode_image(self, image_path):
        """
        이미지를 base64로 인코딩
        
        Args:
            image_path: 이미지 파일 경로
            
        Returns:
            base64 인코딩된 이미지 문자열
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def analyze_image(self, image_path):
        """
        이미지를 분석하여 한국 관광지 또는 음식 정보 제공
        
        Args:
            image_path: 분석할 이미지 파일 경로
            
        Returns:
            AI의 분석 결과 (dict)
        """
        # 이미지 파일 존재 확인
        if not Path(image_path).exists():
            return {
                "error": f"이미지 파일을 찾을 수 없습니다: {image_path}"
            }
        
        # 이미지 인코딩
        base64_image = self.encode_image(image_path)
        
        # GPT-4 Vision API 호출
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": """당신은 한국의 관광지와 음식 전문가입니다. 
                        이미지를 분석하여 JSON 형식으로 정보를 제공하세요.
                        
                        음식인 경우:
                        {
                            "type": "음식",
                            "음식명": "음식 이름",
                            "대부분_들어가있는_재료": ["재료1", "재료2", "재료3"],
                            "음식에_대한_설명": "상세한 설명",
                            "음식_특징": "특별한 특징이나 맛의 특성"
                        }
                        
                        장소인 경우:
                        {
                            "type": "장소",
                            "장소_이름": "장소 이름",
                            "장소에_대한_설명": "상세한 설명",
                            "장소에_대한_특징": "특별한 특징이나 역사적 의미"
                        }
                        
                        반드시 유효한 JSON 형식으로만 답변하세요."""
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "이 사진은 한국의 어떤 관광지 또는 어떤 음식인가요? JSON 형식으로 자세히 설명해주세요."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=1000,
                response_format={"type": "json_object"}
            )
            
            # JSON 파싱
            analysis_text = response.choices[0].message.content
            analysis_json = json.loads(analysis_text)
            
            result = {
                "success": True,
                "data": analysis_json,
                "model": response.model,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            }
            
            return result
            
        except json.JSONDecodeError as e:
            return {
                "success": False,
                "error": f"JSON 파싱 오류: {str(e)}",
                "raw_response": analysis_text
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"API 호출 중 오류 발생: {str(e)}"
            }
    
    def analyze_image_url(self, image_url):
        """
        URL의 이미지를 분석
        
        Args:
            image_url: 이미지 URL
            
        Returns:
            AI의 분석 결과 (dict)
        """
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": """당신은 한국의 관광지와 음식 전문가입니다. 
                        이미지를 분석하여 JSON 형식으로 정보를 제공하세요.
                        
                        음식인 경우:
                        {
                            "type": "음식",
                            "음식명": "음식 이름",
                            "대부분_들어가있는_재료": ["재료1", "재료2", "재료3"],
                            "음식에_대한_설명": "상세한 설명",
                            "음식_특징": "특별한 특징이나 맛의 특성"
                        }
                        
                        장소인 경우:
                        {
                            "type": "장소",
                            "장소_이름": "장소 이름",
                            "장소에_대한_설명": "상세한 설명",
                            "장소에_대한_특징": "특별한 특징이나 역사적 의미"
                        }
                        
                        반드시 유효한 JSON 형식으로만 답변하세요."""
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "이 사진은 한국의 어떤 관광지 또는 어떤 음식인가요? JSON 형식으로 자세히 설명해주세요."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": image_url
                                }
                            }
                        ]
                    }
                ],
                max_tokens=1000,
                response_format={"type": "json_object"}
            )
            
            # JSON 파싱
            analysis_text = response.choices[0].message.content
            analysis_json = json.loads(analysis_text)
            
            result = {
                "success": True,
                "data": analysis_json,
                "model": response.model,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            }
            
            return result
            
        except json.JSONDecodeError as e:
            return {
                "success": False,
                "error": f"JSON 파싱 오류: {str(e)}",
                "raw_response": analysis_text
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"API 호출 중 오류 발생: {str(e)}"
            }


def main():
    """사용 예시"""
    
    # API 키 설정 (환경 변수 또는 직접 입력)
    # export OPENAI_API_KEY='your-api-key-here'
    
    print("=" * 60)
    print("한국 관광지 및 음식 이미지 분석 AI")
    print("=" * 60)
    
    try:
        analyzer = KoreanImageAnalyzer()
        
        # 사용자 입력
        print("\n1. 로컬 이미지 파일 분석")
        print("2. URL 이미지 분석")
        choice = input("\n선택하세요 (1 or 2): ").strip()
        
        if choice == "1":
            image_path = input("이미지 파일 경로를 입력하세요: ").strip()
            print("\n이미지 분석 중...")
            result = analyzer.analyze_image(image_path)
        elif choice == "2":
            image_url = input("이미지 URL을 입력하세요: ").strip()
            print("\n이미지 분석 중...")
            result = analyzer.analyze_image_url(image_url)
        else:
            print("잘못된 선택입니다.")
            return
        
        # 결과 출력
        print("\n" + "=" * 60)
        if result.get("success"):
            print("분석 결과 (JSON):")
            print("-" * 60)
            print(json.dumps(result["data"], ensure_ascii=False, indent=2))
            print("\n" + "-" * 60)
            print(f"모델: {result['model']}")
            print(f"사용된 토큰: {result['usage']['total_tokens']}")
        else:
            print("오류:", result.get("error"))
            if "raw_response" in result:
                print("\n원본 응답:")
                print(result["raw_response"])
        print("=" * 60)
        
    except ValueError as e:
        print(f"\n오류: {e}")
        print("\nOpenAI API 키를 설정하는 방법:")
        print("1. .env 파일 생성:")
        print("   cp .env.example .env")
        print("   그리고 .env 파일에 실제 API 키 입력")
        print("\n2. 또는 환경 변수로 설정:")
        print("   export OPENAI_API_KEY='your-api-key-here'")
        print("\n현재 API 키 상태:", "설정됨" if os.getenv('OPENAI_API_KEY') else "설정 안 됨")
    except Exception as e:
        print(f"\n예상치 못한 오류: {e}")


if __name__ == "__main__":
    main()
