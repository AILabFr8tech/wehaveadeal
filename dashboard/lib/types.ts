// Mirrors wehaveadeal/models.py exactly (pydantic model_dump_json field names).

export type Action = "offer" | "accept" | "reject";

export interface TurnRecord {
  turn: number;
  actor: "shipper" | "carrier";
  action: Action;
  amount: number | null;
}

export interface TableResult {
  shipper_id: string;
  carrier_id: string;
  closed: boolean;
  impasse: boolean;
  rate: number | null;
  turns_to_close: number | null;
  history: TurnRecord[];
}

export interface MatchEntry {
  shipper_id: string;
  carrier_id: string;
  rate: number;
  turns_to_close: number;
}

export interface MatchingResult {
  matches: MatchEntry[];
  unmatched_shippers: string[];
  shipper_preferences: Record<string, string[]>;
  carrier_preferences: Record<string, string[]>;
}

export interface ShipperNegotiationOutcome {
  shipper_id: string;
  matched: boolean;
  carrier_id: string | null;
  rate: number | null;
  profit: number;
  declined_to_ship: boolean;
  points: number;
}

export interface CarrierNegotiationOutcome {
  carrier_id: string;
  matched: boolean;
  shipper_id: string | null;
  rate: number | null;
  margin: number;
}

export interface RouteInfo {
  route_id: string;
  refprice_mean: number;
  refprice_sigma: number;
}

export interface NegotiationResult {
  negotiation_index: number;
  seed: number;
  sale_price: number;
  goods_cost: number;
  routes: Record<string, RouteInfo>;
  tables: TableResult[];
  matching: MatchingResult;
  shipper_outcomes: ShipperNegotiationOutcome[];
  carrier_outcomes: CarrierNegotiationOutcome[];
}

export interface ShipperStanding {
  points: number;
  profit: number;
  closes: number;
  matches: number;
  got_bonus?: boolean;
}

export interface CarrierStanding {
  margin: number;
  matches: number;
  negotiations: number;
  occupancy: number;
}

export interface RoundResult {
  round_index: number;
  seed: number;
  mode: "frozen" | "learning";
  shipper_ids: string[];
  carrier_ids: string[];
  negotiations: NegotiationResult[];
  shipper_standings: Record<string, ShipperStanding>;
  carrier_standings: Record<string, CarrierStanding>;
  round_winner_shipper: string | null;
  round_winner_carrier: string | null;
}

export type EngineArchitecture = "R" | "RV" | "IS" | "OPT" | "OPT2" | "ML" | "ML2" | "RL" | "RL2";

/** "carrier-opt2-1" -> "OPT2", "shipper-rv-1" -> "RV" (from the id
 * convention shared by both roles: <role>-<kind>-<n>, kind in {r, rv, is,
 * opt, opt2, ml, ml2, rl, rl2} -- see wehaveadeal/simulate.py::build_carrier / build_shipper).
 * Order matters: check the more specific "-opt2-"/"-ml2-"/"-rl2-"/"-rv-"
 * before the bare "-opt-"/"-ml-"/"-rl-"/"-r-", since e.g. "carrier-opt2-1"
 * does NOT contain "-opt-" as a substring but "carrier-rv-1" would wrongly
 * match "-r-" if checked first. */
export function engineLabelForAgent(agentId: string): EngineArchitecture | string {
  // Community agents: "<role>-u_<name>-<n>" (see wehaveadeal/toolkit/loader.py::kind_for).
  // Checked first; their kind never contains "-", so it cannot shadow a baseline match.
  const community = agentId.match(/-(u_[a-z0-9_]+)-\d+$/);
  if (community) return community[1].slice(2).toUpperCase();
  if (agentId.includes("-opt2-")) return "OPT2";
  if (agentId.includes("-opt-")) return "OPT";
  if (agentId.includes("-is-")) return "IS";
  if (agentId.includes("-rv-")) return "RV";
  if (agentId.includes("-rl2-")) return "RL2";
  if (agentId.includes("-rl-")) return "RL";
  if (agentId.includes("-ml2-")) return "ML2";
  if (agentId.includes("-ml-")) return "ML";
  if (agentId.includes("-r-")) return "R";
  return agentId;
}
