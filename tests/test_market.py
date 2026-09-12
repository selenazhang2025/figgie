from itertools import count

from figgie.market import Order, OrderBook, Side

_ids = count()


def order(player, side, price):
    i = next(_ids)
    return Order(order_id=i, player=player, suit=0, side=side, price=price, seq=i)


def test_better_price_fills_first():
    book = OrderBook(0)
    book.submit(order(1, Side.SELL, 9))
    cheap = order(2, Side.SELL, 7)
    book.submit(cheap)
    fill, _ = book.submit(order(3, Side.BUY, 10))
    assert fill.maker == cheap
    assert fill.price == 7  # prints at the resting price, not the taker's limit


def test_earlier_order_fills_first_at_same_price():
    book = OrderBook(0)
    first = order(1, Side.BUY, 5)
    second = order(2, Side.BUY, 5)
    book.submit(first)
    book.submit(second)
    fill, _ = book.submit(order(3, Side.SELL, 5))
    assert fill.maker == first and fill.buyer == 1 and fill.seller == 3
    fill, _ = book.submit(order(3, Side.SELL, 4))
    assert fill.maker == second


def test_non_crossing_order_rests_and_book_stays_sorted():
    book = OrderBook(0)
    for price in (3, 6, 4, 6, 1):
        book.submit(order(1, Side.BUY, price))
    for price in (9, 7, 12, 7):
        book.submit(order(2, Side.SELL, price))
    assert [o.price for o in book.bids] == [6, 6, 4, 3, 1]
    assert [o.seq for o in book.bids[:2]] == sorted(o.seq for o in book.bids[:2])
    assert [o.price for o in book.asks] == [7, 7, 9, 12]
    assert len(book) == 9


def test_ioc_never_rests():
    book = OrderBook(0)
    fill, _ = book.submit(order(1, Side.BUY, 5), ioc=True)
    assert fill is None and len(book) == 0


def test_cancel():
    book = OrderBook(0)
    o = order(1, Side.SELL, 8)
    book.submit(o)
    assert book.cancel(o.order_id) == o
    assert book.cancel(o.order_id) is None
    assert book.best(Side.SELL) is None


def test_self_trade_prevention_cancels_resting_and_continues():
    book = OrderBook(0)
    mine = order(1, Side.SELL, 5)
    theirs = order(2, Side.SELL, 6)
    book.submit(mine)
    book.submit(theirs)
    fill, cancelled = book.submit(order(1, Side.BUY, 6))
    assert cancelled == [mine]
    assert fill.maker == theirs


def test_best_excluding_player():
    book = OrderBook(0)
    book.submit(order(1, Side.BUY, 9))
    other = order(2, Side.BUY, 8)
    book.submit(other)
    assert book.best(Side.BUY, exclude_player=1) == other
