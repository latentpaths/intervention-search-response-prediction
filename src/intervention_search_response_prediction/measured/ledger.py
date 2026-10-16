"""Reconcile persistent request charges without losing unresolved attempts."""


def reconcile(events):
    """Return charges and pending reservations in original event order.

    A response resolves only the most recent reservation for its logical call ID.
    Earlier unanswered attempts remain charged conservatively across restarts.
    """
    spent = {"smoke": 0.0, "study": 0.0}
    pending = {}
    for event in events:
        call_id = event["call_id"]
        if event["event"] == "reserve":
            pending.setdefault(call_id, []).append(event["reserve_usd"])
            spent[event["phase"]] += event["reserve_usd"]
        elif event["event"] in ("response", "error"):
            attempts = pending.get(call_id, [])
            if not attempts:
                raise ValueError("ledger response lacks reservation")
            spent[event["phase"]] += event["charged_usd"] - attempts.pop()
    return spent, pending
