import React from "react";
import { EmotionProps } from "./emotions_interfaces";

export const Emotions: React.FC<EmotionProps> = (props: EmotionProps) => {
	return <div>{props.emotions.length}</div>;
};
