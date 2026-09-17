function [y] = Scale(x,a,b)
y(1,1,:)=normalize(squeeze(x(1,1,:)),'range', [a b]);
y(1,2,:)=normalize(squeeze(x(1,2,:)),'range', [a b]);
y(1,3,:)=normalize(squeeze(x(1,3,:)),'range', [a b]);
y(1,4,:)=normalize(squeeze(x(1,4,:)),'range', [a b]);

y(2,1,:)=normalize(squeeze(x(2,1,:)),'range', [a b]);
y(2,2,:)=normalize(squeeze(x(2,2,:)),'range', [a b]);
y(2,3,:)=normalize(squeeze(x(2,3,:)),'range', [a b]);
y(2,4,:)=normalize(squeeze(x(2,4,:)),'range', [a b]);

y(3,1,:)=normalize(squeeze(x(3,1,:)),'range', [a b]);
y(3,2,:)=normalize(squeeze(x(3,2,:)),'range', [a b]);
y(3,3,:)=normalize(squeeze(x(3,3,:)),'range', [a b]);
y(3,4,:)=normalize(squeeze(x(3,4,:)),'range', [a b]);

y(4,1,:)=normalize(squeeze(x(4,1,:)),'range', [a b]);
y(4,2,:)=normalize(squeeze(x(4,2,:)),'range', [a b]);
y(4,3,:)=normalize(squeeze(x(4,3,:)),'range', [a b]);
y(4,4,:)=normalize(squeeze(x(4,4,:)),'range', [a b]);

y(5,1,:)=normalize(squeeze(x(5,1,:)),'range', [a b]);
y(5,2,:)=normalize(squeeze(x(5,2,:)),'range', [a b]);
y(5,3,:)=normalize(squeeze(x(5,3,:)),'range', [a b]);
y(5,4,:)=normalize(squeeze(x(5,4,:)),'range', [a b]);

y(6,1,:)=normalize(squeeze(x(6,1,:)),'range', [a b]);
y(6,2,:)=normalize(squeeze(x(6,2,:)),'range', [a b]);
y(6,3,:)=normalize(squeeze(x(6,3,:)),'range', [a b]);
y(6,4,:)=normalize(squeeze(x(6,4,:)),'range', [a b]);
end