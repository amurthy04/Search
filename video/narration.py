"""Narration script for the Search explainer video. One entry per narrated segment.
Keys are referenced by the Manim scenes; the text is written for text-to-speech (numbers spelled out where a
model might stumble)."""

SEGMENTS = {
# ---------- 1. intro ----------
"n1": "Here's a question. How do you take hundreds of thousands of web pages, and answer an arbitrary question about them in under two milliseconds? Not a keyword lookup. A real question, in plain English, matched by meaning.",
"n2": "This is the system we built to do that. Four stages. Crawl the pages. Cut them into passages. Turn each passage into a vector. And arrange the vectors into a graph you can navigate. Let's take them one at a time, and look at where the time actually goes.",
# ---------- 2. crawl ----------
"n3": "Fetching a web page is mostly waiting. You resolve a name, open a connection, send a request... and then nothing, for about a second, until the server starts replying. The download itself is a rounding error.",
"n4": "So a crawler never does one thing at a time. A single process keeps hundreds of requests in flight on one event loop. Whenever one request is waiting, the loop just moves on to the next.",
"n5": "Here's the catch we ran into. Nearly every link points at a host we've never seen before. Each one needs a brand new connection, and our network path only allows about a hundred and twenty new connections per second. Push past that, and connections start queuing upstream. Handshakes that took a tenth of a second start taking seven.",
"n6": "The fix is in choosing which wait to start next. Hosts we've already talked to are warm. They have an open connection, so the next page costs nothing to set up. Hosts we've never seen are cold. Serve the warm ones first, and let the cold ones through a token bucket, at a fixed rate.",
"n7": "Same minute, same starting URLs. Fifty two pages a second without it. Three hundred and twenty one with it. At that point the ceiling had moved out of the code, and into the network.",
# ---------- 3. passages ----------
"n8": "Next, passages. An embedding model reads a window of about a hundred and eighty words and compresses it into a single vector. Give it a whole page, and you get a blurry average of everything on it. So each page is sliced into overlapping windows, and the page's title is stapled onto the front of every one. That way a paragraph buried deep in an article still knows what the article is about.",
# ---------- 4. embed ----------
"n9": "Now the central idea. Each passage goes through a small transformer, twelve layers, thirty three million parameters, and comes out as a list of three hundred and eighty four numbers. Then we scale that list so its length is exactly one.",
"n10": "That turns every passage into a point on the surface of a sphere. And the model was trained so that a question, and the passage that answers it, point in nearly the same direction.",
"n11": "Which means similarity is just an angle. The dot product of two unit vectors is the cosine of the angle between them. One when they agree. Zero when they have nothing to do with each other.",
"n12": "On real queries, it looks like this. How do m R N A vaccines train the immune system, scores zero point eight seven five against a Scientific American explainer. A question about the Treaty of Westphalia, which the crawl never reached, tops out at zero point seven four. Every correct answer scored above point eight. Every miss, below point seven five.",
"n13": "Doing this two and a half million times is the heaviest compute in the whole project. The stock model ran at about seven hundred passages a second on this laptop. A hand written copy of the model with fixed shapes, compiled for the GPU, ran at fifteen hundred. And we never trained anything. The model's published accuracy reproduces exactly, so the model was never the bottleneck.",
# ---------- 5. hnsw ----------
"n14": "So now we have two and a half million points on a sphere, and a query. Which ten are closest? The honest way is to check every single one. That takes sixteen milliseconds here, and it grows with every page you add.",
"n15": "Instead, connect each point to its nearest neighbours, and search by walking. Start anywhere. Look at your neighbours. Step to whichever one is closest to the query. Stop when no neighbour gets you any closer.",
"n16": "A flat graph has a problem though. If you start far away, you take a lot of small steps. So we stack layers, like a skip list. Every point lives on the bottom layer. Each one gets promoted upward with a small probability, so the top layer has just a few points, with long range links. Search enters at the top, walks to the closest point there, drops down a layer, and zooms in.",
"n17": "On the bottom layer, instead of one best point, we keep a list of candidates. Its size is called e f. A bigger e f explores more neighbours. Better recall, more time. At e f sixty four, we find ninety nine point one percent of the true top ten, in zero point one three milliseconds. That's a hundred and twenty times faster than checking everything, for less than one percent of recall.",
# ---------- 6. query ----------
"n18": "Put it all together, and here is one query, in slow motion. The question is embedded with the same model. One point six eight milliseconds. The graph walk, zero point one nine. Then collapsing duplicates, syndicated copies of the same article, and reading snippets out of a memory mapped file. A twentieth of a millisecond, between them.",
"n19": "Total, one point nine four milliseconds. Eighty five percent of it is the encoder. The index is almost free.",
# ---------- 7. outro ----------
"n20": "Every stage was pushed until it stopped moving. The crawler hit the network path. The embedder hit the GPU. The index is at ninety nine percent recall. And the query is one transformer pass away from the floor. The only lever left is a smaller model on the query side, which is the one place where training would finally pay off.",
"n21": "And that's the whole system. Pages, to passages, to directions on a sphere, to a graph you can walk in a millisecond.",
}
