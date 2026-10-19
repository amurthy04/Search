"""Compare DNS resolver strategies: wall time, success, and event-loop blocking."""
import asyncio, sys, time, socket
import aiohttp
import uvloop

hosts = []
with open("data/seeds/top-1m.csv") as f:
    for i, line in enumerate(f):
        if 200000 <= i < 200000 + int(sys.argv[2]):
            hosts.append(line.strip().split(",")[1])


class DnsPyResolver(aiohttp.abc.AbstractResolver):
    def __init__(self, nameservers, concurrency=200):
        import dns.asyncresolver
        self.r = dns.asyncresolver.Resolver(configure=False)
        self.r.nameservers = nameservers
        self.r.lifetime = 4.0
        self.r.timeout = 2.0
        self.sem = asyncio.Semaphore(concurrency)

    async def resolve(self, host, port=0, family=socket.AF_INET):
        import dns.asyncresolver
        async with self.sem:
            ans = await self.r.resolve(host, "A", search=False)
        return [{"hostname": host, "host": r.address, "port": port, "family": socket.AF_INET, "proto": 0, "flags": socket.AI_NUMERICHOST} for r in ans]

    async def close(self):
        pass


async def main(kind):
    if kind == "threaded":
        res = aiohttp.ThreadedResolver()
    elif kind == "aiodns":
        res = aiohttp.AsyncResolver()
    elif kind == "aiodns-public":
        res = aiohttp.AsyncResolver(nameservers=["1.1.1.1", "8.8.8.8", "9.9.9.9"])
    elif kind == "dnspy":
        res = DnsPyResolver(["128.95.112.1", "128.95.120.1"])
    elif kind == "dnspy-public":
        res = DnsPyResolver(["1.1.1.1", "8.8.8.8", "9.9.9.9", "1.0.0.1", "8.8.4.4"])
    lag = []
    async def lagmon():
        while True:
            t = time.perf_counter(); await asyncio.sleep(0.01); lag.append(time.perf_counter() - t - 0.01)
    mon = asyncio.create_task(lagmon())
    sem = asyncio.Semaphore(200)
    ok = 0; fail = 0
    async def one(h):
        nonlocal ok, fail
        async with sem:
            try:
                await asyncio.wait_for(res.resolve(h, 443), 5)
                ok += 1
            except Exception:
                fail += 1
    t0 = time.perf_counter()
    await asyncio.gather(*(one(h) for h in hosts))
    dt = time.perf_counter() - t0
    mon.cancel()
    lag.sort()
    print(f"{kind:14s} hosts={len(hosts)} ok={ok} fail={fail} wall={dt:.1f}s rate={len(hosts)/dt:.0f}/s loop_lag p50={lag[len(lag)//2]*1000:.1f}ms p99={lag[int(len(lag)*.99)]*1000:.1f}ms max={lag[-1]*1000:.0f}ms", flush=True)
    await res.close()

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
asyncio.run(main(sys.argv[1]))
