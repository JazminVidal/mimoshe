import { FC } from "react";

type InfoTipProps = {
  text: string;
};

export const InfoTip: FC<InfoTipProps> = ({ text }) => {
  return (
    <span className="info-tip" tabIndex={0} aria-label={text}>
      ?
      <span role="tooltip">{text}</span>
    </span>
  );
};
