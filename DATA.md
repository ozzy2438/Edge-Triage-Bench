# Data

**Dataset: BANKING77** (fallback). CFPB was the primary choice, but the CFPB stopped publishing complaint narratives ([announcement, 14 Aug 2026](https://www.consumerfinance.gov/about-us/newsroom/the-cfpb-to-cease-discretionary-publication-of-complaint-narratives-and-visualizations/); removed in database Release 24, September 2026); `complaints.csv.zip` (downloaded 2026-09-25) has no narrative column and the API returns none. See `DECISIONS.md`.

- Source: [PolyAI-LDN/task-specific-datasets@9d081458ff](https://github.com/PolyAI-LDN/task-specific-datasets/tree/9d081458ff52e53cf7e848f414e6e9344e4e6696/banking_data) (same files as [PolyAI/banking77](https://huggingface.co/datasets/PolyAI/banking77)), CC-BY-4.0, downloaded 2026-09-25.
- Raw SHA-256: train `b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b`, test `d12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d`.
- Task: classify a customer-service message into one of 8 coarse triage labels.
- `dev` (100) and `test` (500): stratified sample (seed 42) of the official test split after de-duplicating case/whitespace-normalised text.
- `train` (TF-IDF baseline only): official train split minus any text also in dev/test (2 removed) and minus duplicates (4 removed).
- Truncation: messages are capped at 128 tokens per model tokenizer (logged per run); BANKING77 messages are short.

| Split | N | SHA-256 |
|---|---|---|
| train | 9997 | `4e0648808d6d0d81eece5b8984faf80684395f9644f6c93e522824cffbb02060` |
| dev | 100 | `e198c5d5fb5066bc7f48e04e7643bee685d5a77c3ae248bfff1aab158564e49d` |
| test | 500 | `f36a9d49993ff34551135ea219bd4f1309bff90f2b18cd3f4b5c1c06a730af43` |

## Label mapping (77 intents -> 8 labels)

| Label | Test N | Dev N | Intents | Why merged |
|---|---|---|---|---|
| card_setup | 77 | 16 | `activate_my_card` `card_about_to_expire` `card_arrival` `card_delivery_estimate` `card_linking` `get_disposable_virtual_card` `get_physical_card` `getting_spare_card` `getting_virtual_card` `order_physical_card` `disposable_card_limits` `visa_or_mastercard` | All concern getting a card into the customer's hands and working for the first time. |
| card_problem | 77 | 16 | `card_not_working` `contactless_not_working` `virtual_card_not_working` `card_swallowed` `compromised_card` `lost_or_stolen_card` `lost_or_stolen_phone` `pin_blocked` `change_pin` `passcode_forgotten` `card_acceptance` `declined_card_payment` | A card or access credential that should work does not, or is at risk: same fraud/support queue. |
| card_payment | 72 | 14 | `card_payment_fee_charged` `card_payment_not_recognised` `card_payment_wrong_exchange_rate` `pending_card_payment` `reverted_card_payment?` `transaction_charged_twice` `direct_debit_payment_not_recognised` `extra_charge_on_statement` `apple_pay_or_google_pay` `Refund_not_showing_up` `request_refund` | Disputes or questions about a specific purchase or charge: payments/disputes queue. |
| cash_atm | 45 | 9 | `atm_support` `cash_withdrawal_charge` `cash_withdrawal_not_recognised` `declined_cash_withdrawal` `pending_cash_withdrawal` `wrong_amount_of_cash_received` `wrong_exchange_rate_for_cash_withdrawal` | All involve physical cash from an ATM. |
| top_up | 72 | 14 | `automatic_top_up` `pending_top_up` `top_up_by_bank_transfer_charge` `top_up_by_card_charge` `top_up_by_cash_or_cheque` `top_up_failed` `top_up_limits` `top_up_reverted` `topping_up_by_card` `verify_top_up` `balance_not_updated_after_cheque_or_cash_deposit` | All concern funding the account; cheque/cash deposit balance is a top-up by another channel. |
| transfer | 72 | 14 | `balance_not_updated_after_bank_transfer` `beneficiary_not_allowed` `cancel_transfer` `declined_transfer` `failed_transfer` `pending_transfer` `receiving_money` `transfer_fee_charged` `transfer_into_account` `transfer_not_received_by_recipient` `transfer_timing` | All concern account-to-account money movement. |
| exchange | 39 | 8 | `exchange_charge` `exchange_rate` `exchange_via_app` `fiat_currency_support` `country_support` `supported_cards_and_currencies` | FX and geographic/currency availability questions: product-information queue. |
| account | 46 | 9 | `age_limit` `edit_personal_details` `terminate_account` `unable_to_verify_identity` `verify_my_identity` `why_verify_identity` `verify_source_of_funds` | Account-level KYC and lifecycle requests, not tied to a transaction. |

No intents were dropped: every coarse label has at least 30 test items.
