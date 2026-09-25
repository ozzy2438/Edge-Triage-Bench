"""Download BANKING77, map 77 intents to 8 coarse triage labels, build stratified train/dev/test, hash splits."""
import hashlib
import json
import urllib.request

import pandas as pd
from sklearn.model_selection import train_test_split

from etb.configs import DATA, PROCESSED, ROOT, SEED

COMMIT = "9d081458ff52e53cf7e848f414e6e9344e4e6696"
URL = f"https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/{COMMIT}/banking_data/{{}}.csv"

# coarse label -> (description shown to the model, intents, merge justification)
LABELS = {
    "card_setup": ("ordering, delivery, activation, linking or expiry of physical/virtual cards",
                   "activate_my_card card_about_to_expire card_arrival card_delivery_estimate card_linking "
                   "get_disposable_virtual_card get_physical_card getting_spare_card getting_virtual_card "
                   "order_physical_card disposable_card_limits visa_or_mastercard",
                   "All concern getting a card into the customer's hands and working for the first time."),
    "card_problem": ("card not working, declined, lost, stolen, compromised, PIN or passcode issues",
                     "card_not_working contactless_not_working virtual_card_not_working card_swallowed "
                     "compromised_card lost_or_stolen_card lost_or_stolen_phone pin_blocked change_pin "
                     "passcode_forgotten card_acceptance declined_card_payment",
                     "A card or access credential that should work does not, or is at risk: same fraud/support queue."),
    "card_payment": ("card payments, pending/reverted/duplicate charges, unrecognised payments, refunds",
                     "card_payment_fee_charged card_payment_not_recognised card_payment_wrong_exchange_rate "
                     "pending_card_payment reverted_card_payment? transaction_charged_twice "
                     "direct_debit_payment_not_recognised extra_charge_on_statement apple_pay_or_google_pay "
                     "Refund_not_showing_up request_refund",
                     "Disputes or questions about a specific purchase or charge: payments/disputes queue."),
    "cash_atm": ("ATM cash withdrawals, withdrawal fees, declined or wrong cash amounts",
                 "atm_support cash_withdrawal_charge cash_withdrawal_not_recognised declined_cash_withdrawal "
                 "pending_cash_withdrawal wrong_amount_of_cash_received wrong_exchange_rate_for_cash_withdrawal",
                 "All involve physical cash from an ATM."),
    "top_up": ("adding money to the account by card, bank transfer, cash or cheque; top-up limits and failures",
               "automatic_top_up pending_top_up top_up_by_bank_transfer_charge top_up_by_card_charge "
               "top_up_by_cash_or_cheque top_up_failed top_up_limits top_up_reverted topping_up_by_card "
               "verify_top_up balance_not_updated_after_cheque_or_cash_deposit",
               "All concern funding the account; cheque/cash deposit balance is a top-up by another channel."),
    "transfer": ("sending or receiving bank transfers, transfer fees, timing, failures, beneficiaries",
                 "balance_not_updated_after_bank_transfer beneficiary_not_allowed cancel_transfer declined_transfer "
                 "failed_transfer pending_transfer receiving_money transfer_fee_charged transfer_into_account "
                 "transfer_not_received_by_recipient transfer_timing",
                 "All concern account-to-account money movement."),
    "exchange": ("currency exchange rates and charges, supported currencies and countries",
                 "exchange_charge exchange_rate exchange_via_app fiat_currency_support country_support "
                 "supported_cards_and_currencies",
                 "FX and geographic/currency availability questions: product-information queue."),
    "account": ("identity verification, personal details, age limits, closing the account, source of funds",
                "age_limit edit_personal_details terminate_account unable_to_verify_identity verify_my_identity "
                "why_verify_identity verify_source_of_funds",
                "Account-level KYC and lifecycle requests, not tied to a transaction."),
}
INTENT_TO_LABEL = {i: lab for lab, (_, intents, _) in LABELS.items() for i in intents.split()}
LABEL_NAMES = list(LABELS)


def sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_raw(split: str) -> pd.DataFrame:
    path = DATA / "raw" / f"{split}.csv"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(URL.format(split), path)
    df = pd.read_csv(path)
    assert set(df.category) <= set(INTENT_TO_LABEL), set(df.category) - set(INTENT_TO_LABEL)
    df["label"] = df.category.map(INTENT_TO_LABEL)
    df["key"] = df.text.str.lower().str.split().str.join(" ")
    df["id"] = [f"{split}-{i}" for i in range(len(df))]
    return df


def build() -> dict:
    tr, te = load_raw("train"), load_raw("test")
    te = te.drop_duplicates("key")
    pool, _ = train_test_split(te, train_size=600, stratify=te.label, random_state=SEED)
    dev, test = train_test_split(pool, train_size=100, stratify=pool.label, random_state=SEED)
    train = tr[~tr.key.isin(pool.key)].drop_duplicates("key")
    PROCESSED.mkdir(parents=True, exist_ok=True)
    info = {"leaked_train_rows_removed": int(tr.key.isin(pool.key).sum()),
            "train_dupes_removed": int(len(tr[~tr.key.isin(pool.key)]) - len(train))}
    for name, df in [("train", train), ("dev", dev), ("test", test)]:
        path = PROCESSED / f"{name}.jsonl"
        with open(path, "w") as f:
            for r in df.sort_values("id").itertuples():
                f.write(json.dumps({"id": r.id, "text": r.text, "label": r.label, "intent": r.category}) + "\n")
        info[name] = {"n": len(df), "sha256": sha256(path), "counts": df.label.value_counts().to_dict()}
    info["raw_sha256"] = {s: sha256(DATA / "raw" / f"{s}.csv") for s in ("train", "test")}
    return info


def write_md(info: dict) -> None:
    L = ["# Data", "",
         "**Dataset: BANKING77** (fallback). CFPB was the primary choice, but CFPB Release 23 (July 2026) removed "
         "consumer complaint narratives from the public database; `complaints.csv.zip` (downloaded 2026-09-25) "
         "has no narrative column and the API returns none. See `DECISIONS.md`.", "",
         f"- Source: [PolyAI-LDN/task-specific-datasets@{COMMIT[:10]}](https://github.com/PolyAI-LDN/task-specific-datasets/tree/{COMMIT}/banking_data) "
         "(same files as [PolyAI/banking77](https://huggingface.co/datasets/PolyAI/banking77)), CC-BY-4.0, downloaded 2026-09-25.",
         f"- Raw SHA-256: train `{info['raw_sha256']['train']}`, test `{info['raw_sha256']['test']}`.",
         "- Task: classify a customer-service message into one of 8 coarse triage labels.",
         f"- `dev` (100) and `test` (500): stratified sample (seed {SEED}) of the official test split after de-duplicating "
         "case/whitespace-normalised text.",
         f"- `train` (TF-IDF baseline only): official train split minus any text also in dev/test "
         f"({info['leaked_train_rows_removed']} removed) and minus duplicates ({info['train_dupes_removed']} removed).",
         "- Truncation: messages are capped at 128 tokens per model tokenizer (logged per run); BANKING77 messages are short.",
         "", "| Split | N | SHA-256 |", "|---|---|---|"]
    L += [f"| {s} | {info[s]['n']} | `{info[s]['sha256']}` |" for s in ("train", "dev", "test")]
    L += ["", "## Label mapping (77 intents -> 8 labels)", "", "| Label | Test N | Dev N | Intents | Why merged |", "|---|---|---|---|---|"]
    L += [f"| {lab} | {info['test']['counts'].get(lab, 0)} | {info['dev']['counts'].get(lab, 0)} | {' '.join(f'`{i}`' for i in intents.split())} | {why} |"
          for lab, (_, intents, why) in LABELS.items()]
    L += ["", "No intents were dropped: every coarse label has at least 30 test items."]
    (ROOT / "DATA.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    info = build()
    write_md(info)
    print({s: info[s]["counts"] for s in ("dev", "test")}, info["train"]["n"])
