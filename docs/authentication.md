# Account Authentication

## Two-factor authentication
Tradeport requires two-factor authentication (2FA) for every account before the first withdrawal. Supported methods are authenticator apps (TOTP) and hardware security keys (FIDO2). SMS codes are not supported because of SIM-swap risk.

## Password rules
Passwords must be at least 12 characters long and cannot match any of the last 5 passwords used on the account. Passwords never expire, but Tradeport forces a reset if a password appears in a known breach list.

## Lockout policy
After 5 failed login attempts within 15 minutes, the account is locked for 30 minutes. Each further failed attempt during the lock extends it by another 30 minutes. Support agents cannot manually unlock an account early; the customer must wait or reset their password by email.

## Session length
Web sessions expire after 8 hours of inactivity. Mobile app sessions last 30 days but require biometric confirmation to place trades or request withdrawals.

## API keys
API keys are created from the dashboard under Settings > API. Each account can hold at most 10 active API keys. Keys can be restricted to read-only or trading scope; withdrawal scope is never available through API keys.
