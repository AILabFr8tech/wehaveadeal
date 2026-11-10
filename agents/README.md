# Your agents go here

Every `*.py` file in this folder is scanned for agents. Files starting with `_`
are skipped, so put shared helpers in a file like `_utils.py`. An agent is any
`wehaveadeal.DecisionEngine` subclass with a `role` class attribute:

```python
class MyCarrier(DecisionEngine):
    role = "carrier"          # or "shipper"
    engine_name = "MyCarrier"
    def next_move(self, state): ...
```

Start from the template: `wehaveadeal new-agent MyCarrier --role carrier`.

Agents in this folder show up automatically in the local dashboard's live view
(`docker compose up`, or `wehaveadeal serve`), next to the nine reference
architectures. To check what was found, run `wehaveadeal list-agents`.
