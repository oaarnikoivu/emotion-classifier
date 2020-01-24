export enum Emotions {
	ANGER = "pred_anger",
	ANTICIPATION = "pred_anticipation",
	DISGUST = "pred_disgust",
	FEAR = "pred_fear",
	JOY = "pred_joy",
	LOVE = "pred_love",
	OPTIMISM = "pred_optimism",
	PESSIMISM = "pred_pessimism",
	SADNESS = "pred_sadness",
	SURPRISE = "pred_surprise",
	TRUST = "pred_trust"
}

export interface EmotionProps {
	emotion?: string;
	onNewText?: any;
}
