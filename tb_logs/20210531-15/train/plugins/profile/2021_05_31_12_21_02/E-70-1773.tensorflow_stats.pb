"?>
DDeviceIDLE"IDLE1?????ƠBA?????ƠBQ      ??Y      ???Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1{?G??@9{?G??@A{?G??@I{?G??@a?n?p???i?n?p????Unknown?
BHostIDLE"IDLE1w??/?@Aw??/?@aP??2N,??ip5?Q?b???Unknown
}HostMatMul")gradient_tape/sequential_2/dense_4/MatMul(1Zd;?Omn@9Zd;?Omn@AZd;?Omn@IZd;?Omn@a???kbA??i?qC??j???Unknown
?HostResourceApplyAdam"$Adam/Adam/update_2/ResourceApplyAdam(1??????j@9??????j@A??????j@I??????j@a???Z???iD/?
^???Unknown
sHost_FusedMatMul"sequential_2/dense_4/Relu(1?p=
?[_@9?p=
?[_@A?p=
?[_@I?p=
?[_@a?~?????iI?V????Unknown
?HostResourceApplyAdam""Adam/Adam/update/ResourceApplyAdam(1j?t??L@9j?t??L@Aj?t??L@Ij?t??L@a?ob?????i?0?p?b???Unknown
qHostSoftmax"sequential_2/dense_5/Softmax(1ףp=
?7@9ףp=
?7@Aףp=
?7@Iףp=
?7@a,?n`????i?-??????Unknown
v	Host_FusedMatMul"sequential_2/dense_5/BiasAdd(1+???6@9+???6@A+???6@I+???6@aز;H??i?f+?????Unknown

HostMatMul"+gradient_tape/sequential_2/dense_5/MatMul_1(1Zd;?o3@9Zd;?o3@AZd;?o3@IZd;?o3@a???U4?~?i??œ?5???Unknown
^HostGatherV2"GatherV2(133333?2@933333?2@A33333?2@I33333?2@a?7?j~?i?j4i?q???Unknown
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1?"??~*3@9?"??~*3@A'1??0@I'1??0@aw??;#?z?i?ꫯb????Unknown
?Host#SparseSoftmaxCrossEntropyWithLogits"gsparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits(1?G?z./@9?G?z./@A?G?z./@I?G?z./@a??΃?x?i????????Unknown
?HostResourceApplyAdam"$Adam/Adam/update_3/ResourceApplyAdam(1?? ?r?(@9?? ?r?(@A?? ?r?(@I?? ?r?(@a[?Xu?s?i?9??? ???Unknown
}HostMatMul")gradient_tape/sequential_2/dense_5/MatMul(1+?َ$@9+?َ$@A+?َ$@I+?َ$@a???^cp?i????!???Unknown
sHostDataset"Iterator::Model::ParallelMapV2(17?A`?P#@97?A`?P#@A7?A`?P#@I7?A`?P#@a??}??n?i?7??a@???Unknown
iHostWriteSummary"WriteSummary(1V-2#@9V-2#@AV-2#@IV-2#@a???"ךn?i?ۺa?^???Unknown?
?HostResourceApplyAdam"$Adam/Adam/update_1/ResourceApplyAdam(1??/?d"@9??/?d"@A??/?d"@I??/?d"@aw* ?Sm?i????O|???Unknown
?HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1?(\??u%@9?(\??u%@A-?????@I-?????@a?G??h?i?? ~ؔ???Unknown
`HostGatherV2"
GatherV2_1(1??Q??@9??Q??@A??Q??@I??Q??@av?_Y?hh?i?KzCA????Unknown
?HostMul"Ugradient_tape/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/mul(1F?????@9F?????@AF?????@IF?????@a<A?SN?e?i??͑?????Unknown
vHostCast"$sparse_categorical_crossentropy/Cast(1h??|?5@9h??|?5@Ah??|?5@Ih??|?5@a??gQd?i?`5??????Unknown
?HostCast"`sparse_categorical_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_float_Cast(1      @9      @A      @I      @a-F7??`?ix?lu?????Unknown
[HostAddV2"Adam/add(1?z?G?@9?z?G?@A?z?G?@I?z?G?@ag?	?`?i?Cv?<????Unknown
dHostDataset"Iterator::Model(1V-2-@9V-2-@A??(\??@I??(\??@a?`|i	?_?i+????Unknown
?HostSum"1sparse_categorical_crossentropy/weighted_loss/Sum(1????K?@9????K?@A????K?@I????K?@ar#S???]?i???????Unknown
lHostIteratorGetNext"IteratorGetNext(1?Zd;@9?Zd;@A?Zd;@I?Zd;@aâ7??Y?irG??#???Unknown
vHostReadVariableOp"Adam/Cast_3/ReadVariableOp(1?&1?@9?&1?@A?&1?@I?&1?@a~r?]?Y?i+9??0???Unknown
?HostBiasAddGrad"6gradient_tape/sequential_2/dense_5/BiasAdd/BiasAddGrad(1????K7@9????K7@A????K7@I????K7@a???SX?i??H?<???Unknown
?HostReadVariableOp"*sequential_2/dense_4/MatMul/ReadVariableOp(1???K7?@9???K7?@A???K7?@I???K7?@a??3?z?V?iroH???Unknown
?HostBiasAddGrad"6gradient_tape/sequential_2/dense_4/BiasAdd/BiasAddGrad(1??~j?t@9??~j?t@A??~j?t@I??~j?t@a#?l*'?V?i?I?sS???Unknown
? HostReluGrad"+gradient_tape/sequential_2/dense_4/ReluGrad(15^?I@95^?I@A5^?I@I5^?I@a?{y??U?i???l7^???Unknown
x!HostDataset"#Iterator::Model::ParallelMapV2::Zip(1D?l???@@9D?l???@@A??~j?t
@I??~j?t
@aD?? U?i?????h???Unknown
V"HostSum"Sum_2(1?rh??|	@9?rh??|	@A?rh??|	@I?rh??|	@aT??|uQT?i?座?r???Unknown
?#HostDivNoNan"3sparse_categorical_crossentropy/weighted_loss/value(1??? ?r@9??? ?r@A??? ?r@I??? ?r@a?#??8}S?i׶D?|???Unknown
Z$HostArgMax"ArgMax(1}?5^?I@9}?5^?I@A}?5^?I@I}?5^?I@a???\S?i.???X????Unknown
?%HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice(1}?5^?I@9}?5^?I@A}?5^?I@I}?5^?I@a???\S?i?˼?????Unknown
Y&HostPow"Adam/Pow(1ffffff@9ffffff@Affffff@Iffffff@a?$(z3Q?i??yo?????Unknown
e'Host
LogicalAnd"
LogicalAnd(1ffffff@9ffffff@Affffff@Iffffff@a'?W?N?i???I????Unknown?
o(HostReadVariableOp"Adam/ReadVariableOp(1?E????@9?E????@A?E????@I?E????@a??FN?i?j??˧???Unknown
?)HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1P??n?@9P??n???AP??n?@IP??n???a?7w???M?i??7r,????Unknown
t*HostAssignAddVariableOp"AssignAddVariableOp(1F????x@9F????x@AF????x@IF????x@a+@H?0?K?i?g>#????Unknown
[+HostPow"
Adam/Pow_1(1?????M??9?????M??A?????M??I?????M??a`?A?H(H?i(??P-????Unknown
v,HostAssignAddVariableOp"AssignAddVariableOp_2(1+??????9+??????A+??????I+??????a?y2ж?G?i??;>&????Unknown
~-HostAssignAddVariableOp"Adam/Adam/AssignAddVariableOp(1??ʡE??9??ʡE??A??ʡE??I??ʡE??aa巁'?A?i?e??????Unknown
X.HostEqual"Equal(1??ʡE??9??ʡE??A??ʡE??I??ʡE??aa巁'?A?i????????Unknown
b/HostDivNoNan"div_no_nan_1(1y?&1???9y?&1???Ay?&1???Iy?&1???aHN1v?z@?i`?z%????Unknown
v0HostAssignAddVariableOp"AssignAddVariableOp_4(1?I+???9?I+???A?I+???I?I+???a?d?r?]@?i%w?<????Unknown
?1HostReadVariableOp"+sequential_2/dense_5/BiasAdd/ReadVariableOp(1?O??n??9?O??n??A?O??n??I?O??n??a]nU?:h>?iӰ??	????Unknown
]2HostCast"Adam/Cast_1(1????x???9????x???A????x???I????x???aX?:?&>?i#xr??????Unknown
?3HostCast"bsparse_categorical_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast_1(1?S㥛???9?S㥛???A?S㥛???I?S㥛???aԯ?&?=?i9YR4?????Unknown
T4HostMul"Mul(1?x?&1??9?x?&1??A?x?&1??I?x?&1??a??Uvt?9?i?#?"?????Unknown
X5HostCast"Cast_3(1??C?l??9??C?l??A??C?l??I??C?l??a?& ??9?i?'???????Unknown
v6HostReadVariableOp"Adam/Cast_2/ReadVariableOp(1?MbX9??9?MbX9??A?MbX9??I?MbX9??a????6?i]JԷ?????Unknown
v7HostAssignAddVariableOp"AssignAddVariableOp_1(1?MbX9??9?MbX9??A?MbX9??I?MbX9??a????6?i?l??????Unknown
v8HostAssignAddVariableOp"AssignAddVariableOp_3(1??x?&1??9??x?&1??A??x?&1??I??x?&1??a??)?Fy6?i????N????Unknown
X9HostCast"Cast_2(1/?$????9/?$????A/?$????I/?$????a?"?"3?5?i????????Unknown
?:HostReadVariableOp"+sequential_2/dense_4/BiasAdd/ReadVariableOp(1???Mb??9???Mb??A???Mb??I???Mb??a~	?<?.3?i=?W\t????Unknown
`;HostDivNoNan"
div_no_nan(1m???????9m???????Am???????Im???????a[????e1?iQ???????Unknown
t<HostReadVariableOp"Adam/Cast/ReadVariableOp(1?E??????9?E??????A?E??????I?E??????a?þ]I?0?i)??|?????Unknown
?=HostReadVariableOp"*sequential_2/dense_5/MatMul/ReadVariableOp(1Zd;?O???9Zd;?O???AZd;?O???IZd;?O???al/?B$,/?i????????Unknown
y>HostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1o??ʡ??9o??ʡ??Ao??ʡ??Io??ʡ??a0+c?~,?i??i????Unknown
u?HostReadVariableOp"div_no_nan/ReadVariableOp(1'1?Z??9'1?Z??A'1?Z??I'1?Z??aӋ*?i|?(
????Unknown
w@HostReadVariableOp"div_no_nan/ReadVariableOp_1(1q=
ףp??9q=
ףp??Aq=
ףp??Iq=
ףp??a?u?m?w'?iC?j??????Unknown
wAHostReadVariableOp"div_no_nan_1/ReadVariableOp(1?l??????9?l??????A?l??????I?l??????a?L???z%?i?	?P?????Unknown
aBHostIdentity"Identity(1??v????9??v????A??v????I??v????anHd??j"?i?????????Unknown?*?=
uHostFlushSummaryWriter"FlushSummaryWriter(1{?G??@9{?G??@A{?G??@I{?G??@a?Rc????i?Rc?????Unknown?
}HostMatMul")gradient_tape/sequential_2/dense_4/MatMul(1Zd;?Omn@9Zd;?Omn@AZd;?Omn@IZd;?Omn@a?2?#?P??i???????Unknown
?HostResourceApplyAdam"$Adam/Adam/update_2/ResourceApplyAdam(1??????j@9??????j@A??????j@I??????j@a?҆ ????iA???????Unknown
sHost_FusedMatMul"sequential_2/dense_4/Relu(1?p=
?[_@9?p=
?[_@A?p=
?[_@I?p=
?[_@ah/?Xر?iC}?&????Unknown
?HostResourceApplyAdam""Adam/Adam/update/ResourceApplyAdam(1j?t??L@9j?t??L@Aj?t??L@Ij?t??L@a?ڈA/a??i?
?9????Unknown
qHostSoftmax"sequential_2/dense_5/Softmax(1ףp=
?7@9ףp=
?7@Aףp=
?7@Iףp=
?7@a?v?????i???x/C???Unknown
vHost_FusedMatMul"sequential_2/dense_5/BiasAdd(1+???6@9+???6@A+???6@I+???6@al?q???iÿ󈖫???Unknown
HostMatMul"+gradient_tape/sequential_2/dense_5/MatMul_1(1Zd;?o3@9Zd;?o3@AZd;?o3@IZd;?o3@aE???+??i?~.7???Unknown
^	HostGatherV2"GatherV2(133333?2@933333?2@A33333?2@I33333?2@a?
v??l??i?Vh??Y???Unknown
?
HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1?"??~*3@9?"??~*3@A'1??0@I'1??0@a?x%? .??i??D?~????Unknown
?Host#SparseSoftmaxCrossEntropyWithLogits"gsparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits(1?G?z./@9?G?z./@A?G?z./@I?G?z./@a~(T?N???i`=??w????Unknown
?HostResourceApplyAdam"$Adam/Adam/update_3/ResourceApplyAdam(1?? ?r?(@9?? ?r?(@A?? ?r?(@I?? ?r?(@a????3Y|?i?PK*&???Unknown
}HostMatMul")gradient_tape/sequential_2/dense_5/MatMul(1+?َ$@9+?َ$@A+?َ$@I+?َ$@aЂ??ew?i^"A??T???Unknown
sHostDataset"Iterator::Model::ParallelMapV2(17?A`?P#@97?A`?P#@A7?A`?P#@I7?A`?P#@a.&wa??u?i???????Unknown
iHostWriteSummary"WriteSummary(1V-2#@9V-2#@AV-2#@IV-2#@a??????u?i*????????Unknown?
?HostResourceApplyAdam"$Adam/Adam/update_1/ResourceApplyAdam(1??/?d"@9??/?d"@A??/?d"@I??/?d"@a???H?t?i?F?)~????Unknown
?HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1?(\??u%@9?(\??u%@A-?????@I-?????@aKn?s?q?i?#:?????Unknown
`HostGatherV2"
GatherV2_1(1??Q??@9??Q??@A??Q??@I??Q??@a????lq?i?-	?^???Unknown
?HostMul"Ugradient_tape/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/mul(1F?????@9F?????@AF?????@IF?????@ac\?둲n?i???;???Unknown
vHostCast"$sparse_categorical_crossentropy/Cast(1h??|?5@9h??|?5@Ah??|?5@Ih??|?5@a?n?E??l?in?:??W???Unknown
?HostCast"`sparse_categorical_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_float_Cast(1      @9      @A      @I      @a2z????g?i?.3D?o???Unknown
[HostAddV2"Adam/add(1?z?G?@9?z?G?@A?z?G?@I?z?G?@aԓ
b??g?i|9??k????Unknown
dHostDataset"Iterator::Model(1V-2-@9V-2-@A??(\??@I??(\??@a?-ΎG}f?i?$&?????Unknown
?HostSum"1sparse_categorical_crossentropy/weighted_loss/Sum(1????K?@9????K?@A????K?@I????K?@aY?MpMe?iPU??6????Unknown
lHostIteratorGetNext"IteratorGetNext(1?Zd;@9?Zd;@A?Zd;@I?Zd;@a?R6B\yb?i??֛?????Unknown
vHostReadVariableOp"Adam/Cast_3/ReadVariableOp(1?&1?@9?&1?@A?&1?@I?&1?@a/l??eVb?i ?????Unknown
?HostBiasAddGrad"6gradient_tape/sequential_2/dense_5/BiasAdd/BiasAddGrad(1????K7@9????K7@A????K7@I????K7@a?1??1a?iA???7????Unknown
?HostReadVariableOp"*sequential_2/dense_4/MatMul/ReadVariableOp(1???K7?@9???K7?@A???K7?@I???K7?@a?1? =`?i&??u????Unknown
?HostBiasAddGrad"6gradient_tape/sequential_2/dense_4/BiasAdd/BiasAddGrad(1??~j?t@9??~j?t@A??~j?t@I??~j?t@a:??ny1`?i???z?	???Unknown
?HostReluGrad"+gradient_tape/sequential_2/dense_4/ReluGrad(15^?I@95^?I@A5^?I@I5^?I@aK? ?^?i5????Unknown
xHostDataset"#Iterator::Model::ParallelMapV2::Zip(1D?l???@@9D?l???@@A??~j?t
@I??~j?t
@a1d:^?iN$?(???Unknown
V HostSum"Sum_2(1?rh??|	@9?rh??|	@A?rh??|	@I?rh??|	@aQ?S?0]?iN78?6???Unknown
?!HostDivNoNan"3sparse_categorical_crossentropy/weighted_loss/value(1??? ?r@9??? ?r@A??? ?r@I??? ?r@aλ-6-?[?i?d??}D???Unknown
Z"HostArgMax"ArgMax(1}?5^?I@9}?5^?I@A}?5^?I@I}?5^?I@a?? ??[?i4e^PR???Unknown
?#HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice(1}?5^?I@9}?5^?I@A}?5^?I@I}?5^?I@a?? ??[?ixe?]"`???Unknown
Y$HostPow"Adam/Pow(1ffffff@9ffffff@Affffff@Iffffff@azD[X?i5t?Ol???Unknown
e%Host
LogicalAnd"
LogicalAnd(1ffffff@9ffffff@Affffff@Iffffff@a?z??cV?ir??Zw???Unknown?
o&HostReadVariableOp"Adam/ReadVariableOp(1?E????@9?E????@A?E????@I?E????@a?є??nU?i?2Ȑ????Unknown
?'HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1P??n?@9P??n???AP??n?@IP??n???a?k:p?U?iP?e?????Unknown
t(HostAssignAddVariableOp"AssignAddVariableOp(1F????x@9F????x@AF????x@IF????x@a????S?iaZV??????Unknown
[)HostPow"
Adam/Pow_1(1?????M??9?????M??A?????M??I?????M??a@9???>Q?i???+????Unknown
v*HostAssignAddVariableOp"AssignAddVariableOp_2(1+??????9+??????A+??????I+??????aW??#?Q?iFn?ﱧ???Unknown
~+HostAssignAddVariableOp"Adam/Adam/AssignAddVariableOp(1??ʡE??9??ʡE??A??ʡE??I??ʡE??a??`5-YI?im?0;????Unknown
X,HostEqual"Equal(1??ʡE??9??ʡE??A??ʡE??I??ʡE??a??`5-YI?i?~?^????Unknown
b-HostDivNoNan"div_no_nan_1(1y?&1???9y?&1???Ay?&1???Iy?&1???a!?? ?G?i??F@????Unknown
v.HostAssignAddVariableOp"AssignAddVariableOp_4(1?I+???9?I+???A?I+???I?I+???a|V?R]G?iR?T?????Unknown
?/HostReadVariableOp"+sequential_2/dense_5/BiasAdd/ReadVariableOp(1?O??n??9?O??n??A?O??n??I?O??n??a????ҴE?iz2??????Unknown
]0HostCast"Adam/Cast_1(1????x???9????x???A????x???I????x???aik??4?E?iU:K?????Unknown
?1HostCast"bsparse_categorical_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast_1(1?S㥛???9?S㥛???A?S㥛???I?S㥛???a?#?@\E?i??;[=????Unknown
T2HostMul"Mul(1?x?&1??9?x?&1??A?x?&1??I?x?&1??a??}??B?i?? ?????Unknown
X3HostCast"Cast_3(1??C?l??9??C?l??A??C?l??I??C?l??aLl#`??A?i??z?E????Unknown
v4HostReadVariableOp"Adam/Cast_2/ReadVariableOp(1?MbX9??9?MbX9??A?MbX9??I?MbX9??a?l_2?@?iɗǂI????Unknown
v5HostAssignAddVariableOp"AssignAddVariableOp_1(1?MbX9??9?MbX9??A?MbX9??I?MbX9??a?l_2?@?i?/nM????Unknown
v6HostAssignAddVariableOp"AssignAddVariableOp_3(1??x?&1??9??x?&1??A??x?&1??I??x?&1??a9???@?i?l/P????Unknown
X7HostCast"Cast_2(1/?$????9/?$????A/?$????I/?$????a?
??d??i?C?<????Unknown
?8HostReadVariableOp"+sequential_2/dense_4/BiasAdd/ReadVariableOp(1???Mb??9???Mb??A???Mb??I???Mb??an?'TKc;?i?r?4?????Unknown
`9HostDivNoNan"
div_no_nan(1m???????9m???????Am???????Im???????a@ۮ???8?i?(d	?????Unknown
t:HostReadVariableOp"Adam/Cast/ReadVariableOp(1?E??????9?E??????A?E??????I?E??????aH??%??7?i?(??????Unknown
?;HostReadVariableOp"*sequential_2/dense_5/MatMul/ReadVariableOp(1Zd;?O???9Zd;?O???AZd;?O???IZd;?O???a	8`4?@6?in?҂????Unknown
y<HostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1o??ʡ??9o??ʡ??Ao??ʡ??Io??ʡ??a+?A?C4?iQ?H?????Unknown
u=HostReadVariableOp"div_no_nan/ReadVariableOp(1'1?Z??9'1?Z??A'1?Z??I'1?Z??a?8??R?2?iXѣ?X????Unknown
w>HostReadVariableOp"div_no_nan/ReadVariableOp_1(1q=
ףp??9q=
ףp??Aq=
ףp??Iq=
ףp??aUb>???0?i$???p????Unknown
w?HostReadVariableOp"div_no_nan_1/ReadVariableOp(1?l??????9?l??????A?l??????I?l??????a;@Usi?.?ix?F[????Unknown
a@HostIdentity"Identity(1??v????9??v????A??v????I??v????a~t??K*?i?????????Unknown?2Nvidia GPU (Pascal)