import numpy as np
import random

from core.book import Book
from core.participants import Participant
from order import Order


class LOBEnv:
    """
    Simple RL environment for Limit Order Book interaction.
    Actions:
        0 = HOLD
        1 = BUY (marketable limit)
        2 = SELL (marketable limit)
    """

    def __init__(self, max_steps=100):
        self.max_steps = max_steps
        self.current_step = 0

        # Create agent + counterparty
        self.agent = Participant(1, "agent", balance=10_000, volume=0)
        self.other = Participant(2, "market", balance=10_000, volume=10_000)

        self.book = Book(
            name="RL_BOOK",
            participants=[self.agent, self.other]
        )

        self.last_pnl = 0.0

    # ------------------------
    # Environment API
    # ------------------------

    def reset(self):
        self.current_step = 0

        self.agent.balance = 10_000
        self.agent.volume = 0

        self.book = Book(
            name="RL_BOOK",
            participants=[self.agent, self.other]
        )

        self.last_pnl = self._compute_pnl()

        return self._get_state()

    def step(self, action):
        """
        Execute one environment step.
        """
        self.current_step += 1

        # Random background liquidity
        self._inject_market_order()

        # Agent action
        if action == 1:
            self._place_buy()
        elif action == 2:
            self._place_sell()
        # action == 0 → HOLD

        pnl = self._compute_pnl()
        reward = pnl - self.last_pnl
        self.last_pnl = pnl

        done = self.current_step >= self.max_steps

        return self._get_state(), reward, done, {}

    # ------------------------
    # Helper methods
    # ------------------------

    def _get_state(self):
        """
        State vector:
        [best_bid, best_ask, spread, agent_volume, agent_balance]
        """
        bid = self.book.bids.best
        ask = self.book.asks.best
        spread = ask - bid if bid > 0 and ask > 0 else 0.0

        return np.array([
            bid,
            ask,
            spread,
            self.agent.volume,
            self.agent.balance
        ], dtype=np.float32)

    def _compute_pnl(self):
        bid = self.book.bids.best
        ask = self.book.asks.best

        # Safe midprice computation
        if bid > 0 and ask > 0:
            mid = (bid + ask) / 2
        elif bid > 0:
            mid = bid
        elif ask > 0:
            mid = ask
        else:
            mid = 100.0  # fallback reference price

        return self.agent.balance + self.agent.volume * mid

    def _place_buy(self):
        price = self.book.asks.best or 100
        qty = 1

        if self.agent.balance >= price * qty:
            order = Order(
                id=random.randint(1_000_000, 9_999_999),
                owner=self.agent,
                type="BID",
                price=price,
                qty=qty
            )
            self.book.add(order)

    def _place_sell(self):
        price = self.book.bids.best or 100
        qty = 1

        if self.agent.volume >= qty:
            order = Order(
                id=random.randint(1_000_000, 9_999_999),
                owner=self.agent,
                type="ASK",
                price=price,
                qty=qty
            )
            self.book.add(order)

    def _inject_market_order(self):
        """
        Background market liquidity.
        """
        side = random.choice(["BID", "ASK"])
        price = random.randint(95, 105)
        qty = random.randint(1, 3)

        if side == "BID":
            order = Order(
                id=random.randint(1_000_000, 9_999_999),
                owner=self.other,
                type="BID",
                price=price,
                qty=qty
            )
        else:
            if self.other.volume < qty:
                return
            order = Order(
                id=random.randint(1_000_000, 9_999_999),
                owner=self.other,
                type="ASK",
                price=price,
                qty=qty
            )

        self.book.add(order)
