import type { Action, NegotiationResult, RouteInfo } from "./types";

export interface LiveSetupEvent {
  kind: "setup";
  seed: number;
  sale_price: number;
  goods_cost: number;
  shipper_ids: string[];
  carrier_ids: string[];
  routes: Record<string, RouteInfo>;
}

export interface LiveTurnEvent {
  kind: "turn";
  table_id: string;
  shipper_id: string;
  carrier_id: string;
  turn: number;
  actor: "shipper" | "carrier";
  action: Action;
  amount: number | null;
  closed: boolean;
  impasse: boolean;
}

export interface LiveResultEvent {
  kind: "result";
  negotiation: NegotiationResult;
}

export type LiveEvent = LiveSetupEvent | LiveTurnEvent | LiveResultEvent;
