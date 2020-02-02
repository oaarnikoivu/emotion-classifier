import React from "react";
import { EmotionProps, Emotion } from "./emotions_interfaces";

export const Emotions: React.FC<EmotionProps> = (props: EmotionProps) => {
	return (
		<>
			<div style={{ marginTop: 12, marginBottom: 12, fontFamily: "Arial" }}>
				<h5>{props.showTitle ? "Predictions: " : undefined}</h5>
			</div>
			<div>
				{props.emotions.map((e: any, index: number) => {
					let text: string = "";
					switch (e.id) {
						case Emotion.ANGER:
							text = "Anger";
							break;
						case Emotion.ANTICIPATION:
							text = "Anticipation";
							break;
						case Emotion.DISGUST:
							text = "Disgust";
							break;
						case Emotion.FEAR:
							text = "Fear";
							break;
						case Emotion.JOY:
							text = "Joy";
							break;
						case Emotion.LOVE:
							text = "Love";
							break;
						case Emotion.OPTIMISM:
							text = "Optimism";
							break;
						case Emotion.PESSIMISM:
							text = "Pessimism";
							break;
						case Emotion.SADNESS:
							text = "Sadness";
							break;
						case Emotion.SURPRISE:
							text = "Surprise";
							break;
						case Emotion.TRUST:
							text = "Trust";
							break;
					}
					return (
						<p key={e.id + index}>
							{text} -> {Math.round(e.value * 100 + Number.EPSILON) / 100}
						</p>
					);
				})}
			</div>
		</>
	);
};
