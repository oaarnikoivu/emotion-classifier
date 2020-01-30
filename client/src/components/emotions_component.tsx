import React from "react";
import { EmotionProps } from "./emotions_interfaces";
import { Emoji } from "./emoji/emoji_component";

export const Emotions: React.FC<EmotionProps> = (props: EmotionProps) => {
	return (
		<>
			<div>{props.emotion}</div>
			<Emoji symbol={"\u2728"} label='sheep' />
		</>
	);
};
