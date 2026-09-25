# API Rate Limits

## Default limits
The Tradeport REST API allows 120 requests per minute per API key. The WebSocket API allows 50 messages per second per connection and at most 5 concurrent connections per account.

## Order endpoints
Order placement and cancellation share a separate bucket of 20 requests per second per account. Batch orders count as one request per order inside the batch, up to 10 orders per batch.

## What happens when you exceed a limit
Requests over the limit receive HTTP 429 with a Retry-After header in seconds. Clients that keep sending requests while throttled for more than 60 seconds are blocked for 10 minutes. Repeated blocks within 24 hours may lead to the API key being revoked.

## Higher limits
Accounts with a 30-day trading volume above 1,000,000 USD can request a higher tier through support. The higher tier allows 600 requests per minute on REST and 100 order requests per second. Requests are reviewed within 3 business days.
