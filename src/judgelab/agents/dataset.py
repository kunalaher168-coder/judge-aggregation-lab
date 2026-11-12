"""Synthetic support-reply dataset with per-criterion ground truth.

Each item is a store refund policy, an order, a customer message and an
agent's reply. Defects are injected independently per criterion, so the
correct pass/fail label for every criterion is known by construction:

  policy_fidelity          reply states a policy fact that contradicts the policy
  refund_calculation       stated refund amount differs from the correct amount
  no_unsupported_promises  reply offers something the policy explicitly rules out
  professional_tone        reply blames or talks down to the customer

Everything is template-generated from a seed. Names, products and policies
are invented.
"""

from __future__ import annotations

import random
from decimal import ROUND_HALF_UP, Decimal

CRITERIA = ["policy_fidelity", "refund_calculation", "no_unsupported_promises", "professional_tone"]

PRODUCTS = [
    "desk lamp",
    "backpack",
    "electric kettle",
    "bike helmet",
    "phone case",
    "yoga mat",
    "pair of headphones",
    "coffee grinder",
    "rain jacket",
    "blender",
    "office chair",
    "water bottle",
]
CUSTOMERS = [
    "Sam",
    "Priya",
    "Jordan",
    "Mei",
    "Alex",
    "Tomasz",
    "Aisha",
    "Luca",
    "Noor",
    "Daniel",
    "Ines",
    "Kofi",
]
AGENTS = ["Robin", "Casey", "Morgan", "Taylor"]
CENT = Decimal("0.01")


def money(x: Decimal) -> str:
    return f"${x.quantize(CENT, rounding=ROUND_HALF_UP)}"


def _price(rng: random.Random) -> Decimal:
    whole = rng.randint(19, 249)
    cents = rng.choice(["0.99", "0.50", "0.00", "0.95", "0.49"])
    return Decimal(whole) + Decimal(cents)


def make_item(
    rng: random.Random, idx: int, defect_rate: float = 0.35
) -> tuple[dict, dict[str, str]]:
    window = rng.choice([14, 30, 60])
    fee_pct = rng.choice([10, 15, 20])
    label_fee = rng.choice([Decimal("4.95"), Decimal("5.95"), Decimal("7.50"), None])
    product = rng.choice(PRODUCTS)
    customer = rng.choice(CUSTOMERS)
    agent = rng.choice(AGENTS)
    price = _price(rng)
    defective = rng.random() < 0.25
    opened = True if defective else rng.random() < 0.6
    days = rng.randint(1, window - 1)

    fee = (price * fee_pct / 100).quantize(CENT, rounding=ROUND_HALF_UP)
    applies_fee = opened and not defective
    applies_label = label_fee is not None and not defective
    correct = price - (fee if applies_fee else 0) - (label_fee if applies_label else 0)

    policy = (
        f"Returns are accepted within {window} days of delivery. "
        f"Opened items that are not defective carry a {fee_pct}% restocking fee, charged on the item price. "
        + (
            f"Return shipping costs {money(label_fee)}, deducted from the refund, unless the item is defective. "
            if label_fee is not None
            else "Return shipping is free. "
        )
        + "Defective items get a full refund of the item price and a prepaid return label. "
        "Refunds go back to the original payment method within 5-7 business days. "
        "We do not offer store credit, gifts, discounts or shipping upgrades as compensation."
    )
    order = (
        f"Item: {product}, price {money(price)}. Delivered {days} days ago. "
        f"Opened: {'yes' if opened else 'no'}. Customer reports a defect: {'yes' if defective else 'no'}."
    )
    if defective:
        message = rng.choice(
            [
                f"Hi, my {product} stopped working after two days. Can I return it and how much will I get back?",
                f"The {product} I received is faulty. What refund can I expect if I send it back?",
            ]
        )
    else:
        message = rng.choice(
            [
                f"Hi, I'd like to return the {product} I bought. How much will I be refunded?",
                f"I changed my mind about the {product}. If I send it back, what's my refund?",
                f"Can I return my {product}? Please tell me the exact refund amount.",
            ]
        )

    truth = {c: "pass" for c in CRITERIA}
    bad = {c: rng.random() < defect_rate for c in CRITERIA}

    # --- policy facts stated in the reply -------------------------------
    stated_window, stated_pct, stated_shipping_free = window, fee_pct, label_fee is None
    if bad["policy_fidelity"]:
        options = ["window"]
        if applies_fee:
            options.append("fee")
        if not defective:
            options.append("shipping")
        kind = rng.choice(options)
        if kind == "window":
            stated_window = rng.choice([w for w in (7, 14, 21, 30, 45, 60, 90) if w != window])
        elif kind == "fee":
            stated_pct = rng.choice([p for p in (5, 10, 15, 20, 25) if p != fee_pct])
        else:
            stated_shipping_free = not stated_shipping_free
        truth["policy_fidelity"] = "fail"

    # --- refund amount ---------------------------------------------------
    parts = [("price", price)]
    if applies_fee:
        parts.append(("restocking fee", -fee))
    if applies_label:
        parts.append(("return label", -label_fee))
    amount = correct
    if bad["refund_calculation"]:
        variants = ["slip"]
        if applies_fee:
            variants.append("drop_fee")
        if applies_label:
            variants.append("drop_label")
        if defective:
            variants.append("charge_fee")
        kind = rng.choice(variants)
        if kind == "drop_fee":
            parts = [p for p in parts if p[0] != "restocking fee"]
            amount = sum((v for _, v in parts), Decimal(0))
        elif kind == "drop_label":
            parts = [p for p in parts if p[0] != "return label"]
            amount = sum((v for _, v in parts), Decimal(0))
        elif kind == "charge_fee":
            parts.append(("restocking fee", -fee))
            amount = sum((v for _, v in parts), Decimal(0))
        else:
            amount = correct + rng.choice(
                [
                    Decimal("1.00"),
                    Decimal("-1.00"),
                    Decimal("0.10"),
                    Decimal("-0.90"),
                    Decimal("2.50"),
                ]
            )
        if amount == correct:
            amount = correct + Decimal("1.00")
        truth["refund_calculation"] = "fail"

    # --- assemble the reply ----------------------------------------------
    s = []
    if bad["professional_tone"]:
        s.append(
            rng.choice(
                [
                    f"{customer}, this is all explained on our returns page, which you clearly did not read.",
                    f"Hi {customer}. Honestly, you should have checked the details before ordering.",
                    f"{customer}, as I'm sure you could have worked out yourself, the rules are simple.",
                ]
            )
        )
        truth["professional_tone"] = "fail"
    else:
        s.append(
            rng.choice(
                [
                    f"Hi {customer}, thanks for getting in touch.",
                    f"Hello {customer}, thank you for your message.",
                    f"Hi {customer}, sorry to hear the {product} didn't work out.",
                ]
            )
        )
    s.append(
        f"Our policy accepts returns within {stated_window} days of delivery, so you are within the window."
    )
    if defective:
        s.append(f"As the {product} is defective, no restocking fee applies.")
    elif applies_fee:
        s.append(f"Because the {product} has been opened, a {stated_pct}% restocking fee applies.")
    else:
        s.append(f"As the {product} is unopened, there is no restocking fee.")
    if not defective:
        if stated_shipping_free:
            s.append("Return shipping is free.")
        else:
            s.append(
                f"The {money(label_fee or Decimal('5.95'))} return label is deducted from your refund."
            )
    if rng.random() < 0.5 and len(parts) > 1:
        detail = ", ".join(
            f"{'minus ' if v < 0 else ''}the {money(abs(v))} {name}" for name, v in parts
        )
        s.append(f"That works out as {detail}, so your refund will be {money(amount)}.")
    else:
        s.append(f"Your refund will be {money(amount)}.")
    if bad["no_unsupported_promises"]:
        s.append(
            rng.choice(
                [
                    "To make up for the trouble, I've added a $10 store credit to your account.",
                    "I've also upgraded your next order to free express shipping.",
                    "As a goodwill gesture we'll send you a free gift with your next order.",
                    "I can also give you 15% off your next purchase.",
                ]
            )
        )
        truth["no_unsupported_promises"] = "fail"
    elif rng.random() < 0.5:
        s.append(
            rng.choice(
                ["I've emailed you a prepaid return label."]
                if defective
                else [
                    "The refund will go back to your original payment method within 5-7 business days."
                ]
            )
        )
    if bad["professional_tone"]:
        s.append(rng.choice(["Please read the policy next time.", "Hopefully that is clear now."]))
    else:
        s.append(
            rng.choice(
                [
                    "Let me know if there's anything else I can help with.",
                    "Happy to help with anything else.",
                ]
            )
        )
    s.append(f"Best, {agent}")

    item = {
        "item_id": f"reply_{idx:03d}",
        "policy": policy,
        "order": order,
        "customer_message": message,
        "agent_reply": " ".join(s),
    }
    truth["_correct_refund"] = str(correct.quantize(CENT))
    return item, truth


def generate_support_items(
    n: int = 24, seed: int = 7, defect_rate: float = 0.35
) -> tuple[list[dict], list[dict]]:
    """Return (items, truth_rows); truth_rows are {item_id, criterion, label}."""
    rng = random.Random(seed)
    items, truth_rows = [], []
    for i in range(n):
        item, truth = make_item(rng, i, defect_rate)
        items.append(item)
        for c in CRITERIA:
            row = {"item_id": item["item_id"], "criterion": c, "label": truth[c]}
            if c == "refund_calculation":
                row["correct_refund"] = truth["_correct_refund"]
            truth_rows.append(row)
    return items, truth_rows


RUBRIC = """# Rubric: customer-support refund replies

Grade the agent's reply using only the store policy and order details supplied with it.

1. Policy fidelity [policy_fidelity] - Every policy fact the reply states (return window, restocking fee percentage, who pays return shipping) matches the policy text. Fail if any stated fact contradicts the policy.
2. Refund calculation [refund_calculation] - The refund amount the reply gives is exactly correct to the cent for this order under the policy (restocking fee charged on the item price for opened, non-defective items; return label deducted unless defective or shipping is free).
3. No unsupported promises [no_unsupported_promises] - The reply does not offer anything the policy rules out (store credit, gifts, discounts, shipping upgrades). Promises the policy does provide, such as a prepaid label for a defective item, are fine.
4. Professional tone [professional_tone] - The reply is courteous and does not blame, lecture or talk down to the customer.
"""
