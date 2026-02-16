from copy import deepcopy

from core.side import Side
from core.errors import (
    InsufficientFundsError,
    ParticipantAlreadyExistsError,
    NoSuchParameterError,
)





class Book:
    def __init__(self, name, participants, params=None):
        self.__name = str(name)

        # Participant registry
        self.__participants = {p.id: p for p in participants}

        # Parameters
        self.__params = deepcopy(params) if params else {
            "PartialExecution": True,
            "AllowShorting": False,
            "AllowLending": False,
        }

        # Order book sides
        self.__bids = Side("BID")
        self.__asks = Side("ASK")

        self.__LTP = 0

    # ==================== PROPERTIES ====================

    @property
    def name(self):
        return self.__name

    @property
    def participants(self):
        return list(self.__participants.values())

    @property
    def bids(self):
        return self.__bids

    @property
    def asks(self):
        return self.__asks

    @property
    def top(self):
        return self.bids.best, self.asks.best

    @property
    def spread(self):
        if self.bids.best == 0 or self.asks.best == 0:
            return 0
        return abs(self.bids.best - self.asks.best)

    @property
    def depth(self):
        return self.bids.depth, self.asks.depth

    @property
    def volume(self):
        return self.bids.volume, self.asks.volume

    @property
    def LTP(self):
        return self.__LTP

    # ==================== PARAMETERS ====================

    def set_param(self, name, value):
        if name not in self.__params:
            raise NoSuchParameterError()
        self.__params[name] = value

    def get_param(self, name):
        if name not in self.__params:
            raise NoSuchParameterError()
        return self.__params[name]

    # ==================== PARTICIPANTS ====================

    def add_participant(self, participant):
        if participant.id in self.__participants:
            raise ParticipantAlreadyExistsError()
        self.__participants[participant.id] = participant

    # ==================== MATCHING ENGINE ====================

    def __match(self, counter_side, order):
        # Check if order crosses
        if order.type == "BID":
            crosses = order.price >= counter_side.best
        else:  # ASK
            crosses = order.price <= counter_side.best

        if not crosses or counter_side.volume == 0:
            return False

        # Iterate price levels (already sorted)
        for price in list(counter_side.prices):
            level = counter_side.get(price)

            # FIFO inside level
            while order.qty > 0 and level:
                counter_order = level[0]
                trade_qty = min(order.qty, counter_order.qty)

                self.__execute(order, price, trade_qty)
                self.__execute(counter_order, price, trade_qty)

                if counter_order.qty == 0:
                    level.pop(0)

            # Remove empty price level
            if not level:
                counter_side.prices.remove(price)
                counter_side._Side__data.pop(price, None)

            if order.qty == 0:
                return True

        return True

    # ==================== ACCOUNTING ====================

    def __payout(self, side, order, price, qty):
        participant = self.__participants[order.owner.id]

        if side.type == "BID":
            participant.balance -= price * qty
            participant.volume += qty
        else:  # ASK
            participant.balance += price * qty
            participant.volume -= qty

    # ==================== ORDER ENTRY ====================

    def add(self, order):
        owner = self.__participants[order.owner.id]

        if order.type == "BID":
            if order.price * order.qty > owner.balance and not self.__params["AllowLending"]:
                raise InsufficientFundsError()

            self.__match(self.__asks, order)
            if order.qty > 0:
                self.__bids.put(order)

        elif order.type == "ASK":
            if order.qty > owner.volume and not self.__params["AllowShorting"]:
                raise InsufficientFundsError()

            self.__match(self.__bids, order)
            if order.qty > 0:
                self.__asks.put(order)

    # ==================== EXECUTION ====================

    def __execute(self, order, price, qty):
        if order.type == "BID":
            self.__payout(self.__bids, order, price, qty)
        else:
            self.__payout(self.__asks, order, price, qty)

        order.qty -= qty
        self.__LTP = price

    # ==================== CANCEL ====================

    def cancel(self, order_id):
        self.__bids.remove(order_id)
        self.__asks.remove(order_id)

    # ==================== DISPLAY ====================

    def __repr__(self):
        lines = [f"Book: {self.name}", f"LTP: {self.LTP}", "ASKS:"]
        for p in self.asks.prices:
            qty = sum(o.qty for o in self.asks.get(p))
            lines.append(f"{p} -> {qty}")

        lines.append("BIDS:")
        for p in self.bids.prices:
            qty = sum(o.qty for o in self.bids.get(p))
            lines.append(f"{p} -> {qty}")

        return "\n".join(lines)
