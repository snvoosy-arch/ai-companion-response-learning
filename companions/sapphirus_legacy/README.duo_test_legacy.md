# White Duo Test

White companion은 다른 companion과 함께 대화하는 duo 테스트 모드를 지원합니다.

이 문서는 공개용 요약입니다. 실제 봇 ID, 채널 ID, 로컬 실행 프로필은 저장소에 포함하지 않습니다.

## 설정 예시

```env
DUO_ENABLED=true
DUO_PARTNER_BOT_ID=123456789012345678
DUO_CHANNEL_IDS=123456789012345678
DUO_MAX_REPLIES=8
DUO_REPLY_DELAY_SECONDS=1.5
```

## 검증 포인트

- 두 봇이 같은 채널에서 무한 루프에 빠지지 않는지 확인
- turn limit과 delay가 정상 적용되는지 확인
- 한쪽 봇의 출력 guard가 duo 대화에서도 작동하는지 확인

