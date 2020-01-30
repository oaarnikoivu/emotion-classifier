import React from "react";
import { EmotionProps, Emojis } from "./emotions_interfaces";
import { Emoji } from "./emoji/emoji_component";

export const Emotions: React.FC<EmotionProps> = (props: EmotionProps) => {
	return (
		<>
			{/* <div>{props.emotion}</div> */}
			<div>{props.emotion ? determineEmojiToShow() : undefined}</div>
		</>
	);

	function determineEmojiToShow(): JSX.Element {
		let output: JSX.Element;
		switch (props.emotion.toLowerCase()) {
			case "anger":
				output = <Emoji symbol={Emojis.ANGER} label='anger' />;
				break;
			case "anticipation":
				output = <Emoji symbol={Emojis.ANTICIPATION} label='anticipation' />;
				break;
			case "disgust":
				output = <Emoji symbol={Emojis.DISGUST} label='disgust' />;
				break;
			case "fear":
				output = <Emoji symbol={Emojis.FEAR} label='fear' />;
				break;
			case "joy":
				output = <Emoji symbol={Emojis.JOY} label='joy' />;
				break;
			case "love":
				output = <Emoji symbol={Emojis.LOVE} label='love' />;
				break;
			case "optimism":
				output = <Emoji symbol={Emojis.OPTIMISM} label='optimism' />;
				break;
			case "pessimism":
				output = <Emoji symbol={Emojis.PESSIMISM} label='pessimism' />;
				break;
			case "sadness":
				output = <Emoji symbol={Emojis.SADNESS} label='sadness' />;
				break;
			case "surprise":
				output = <Emoji symbol={Emojis.SURPRISE} label='surprise' />;
				break;
			case "trust":
				output = <Emoji symbol={Emojis.TRUST} label='trust' />;
				break;
		}
		return output;
	}
};
