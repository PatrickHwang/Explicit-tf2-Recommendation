†У
иє
D
AddV2
x"T
y"T
z"T"
Ttype:
2	АР
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( И
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(И
?
Mul
x"T
y"T
z"T"
Ttype:
2	Р

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetypeИ
•
ResourceGather
resource
indices"Tindices
output"dtype"

batch_dimsint "
validate_indicesbool("
dtypetype"
Tindicestype:
2	И
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
7
Square
x"T
y"T"
Ttype:
2	
Ѕ
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring И®
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
М
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
Ц
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И"serve*2.8.02v2.8.0-rc1-32-g3f878cff5b68нч
В
fm_ranking_layer/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_namefm_ranking_layer/bias
{
)fm_ranking_layer/bias/Read/ReadVariableOpReadVariableOpfm_ranking_layer/bias*
_output_shapes
:*
dtype0
®
%fm_ranking_layer/embedding/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:
рУ	*6
shared_name'%fm_ranking_layer/embedding/embeddings
°
9fm_ranking_layer/embedding/embeddings/Read/ReadVariableOpReadVariableOp%fm_ranking_layer/embedding/embeddings* 
_output_shapes
:
рУ	*
dtype0
ђ
'fm_ranking_layer/embedding_1/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:
рУ	*8
shared_name)'fm_ranking_layer/embedding_1/embeddings
•
;fm_ranking_layer/embedding_1/embeddings/Read/ReadVariableOpReadVariableOp'fm_ranking_layer/embedding_1/embeddings* 
_output_shapes
:
рУ	*
dtype0

NoOpNoOp
¬
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*э
valueуBр Bй
С
layer-0
layer-1
layer-2
layer-3
layer-4
layer-5
layer-6
layer-7
	layer-8

layer-9
layer-10
layer-11
layer-12
layer_with_weights-0
layer-13
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
њ
feature_names
bias
	embed
w
	variables
trainable_variables
regularization_losses
	keras_api
__call__
* &call_and_return_all_conditional_losses*

0
!1
"2*

0
!1
"2*
* 
∞
#non_trainable_variables

$layers
%metrics
&layer_regularization_losses
'layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 

(serving_default* 
* 
c]
VARIABLE_VALUEfm_ranking_layer/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
†
!
embeddings
)	variables
*trainable_variables
+regularization_losses
,	keras_api
-__call__
*.&call_and_return_all_conditional_losses*
†
"
embeddings
/	variables
0trainable_variables
1regularization_losses
2	keras_api
3__call__
*4&call_and_return_all_conditional_losses*

0
!1
"2*

0
!1
"2*

50
61* 
У
7non_trainable_variables

8layers
9metrics
:layer_regularization_losses
;layer_metrics
	variables
trainable_variables
regularization_losses
__call__
* &call_and_return_all_conditional_losses
& "call_and_return_conditional_losses*
* 
* 
e_
VARIABLE_VALUE%fm_ranking_layer/embedding/embeddings&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUE'fm_ranking_layer/embedding_1/embeddings&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
* 
j
0
1
2
3
4
5
6
7
	8

9
10
11
12
13*
* 
* 
* 
* 

!0*

!0*
	
50* 
У
<non_trainable_variables

=layers
>metrics
?layer_regularization_losses
@layer_metrics
)	variables
*trainable_variables
+regularization_losses
-__call__
*.&call_and_return_all_conditional_losses
&."call_and_return_conditional_losses*
* 
* 

"0*

"0*
	
60* 
У
Anon_trainable_variables

Blayers
Cmetrics
Dlayer_regularization_losses
Elayer_metrics
/	variables
0trainable_variables
1regularization_losses
3__call__
*4&call_and_return_all_conditional_losses
&4"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

0
1*
* 
* 
* 
* 
* 
* 
	
50* 
* 
* 
* 
* 
	
60* 
* 
v
serving_default_iidPlaceholder*'
_output_shapes
:€€€€€€€€€*
dtype0	*
shape:€€€€€€€€€
x
serving_default_itag1Placeholder*'
_output_shapes
:€€€€€€€€€*
dtype0	*
shape:€€€€€€€€€
x
serving_default_itag2Placeholder*'
_output_shapes
:€€€€€€€€€*
dtype0	*
shape:€€€€€€€€€
x
serving_default_itag3Placeholder*'
_output_shapes
:€€€€€€€€€*
dtype0	*
shape:€€€€€€€€€
x
serving_default_itag4Placeholder*'
_output_shapes
:€€€€€€€€€*
dtype0	*
shape:€€€€€€€€€
}
serving_default_itag4_cubePlaceholder*'
_output_shapes
:€€€€€€€€€*
dtype0*
shape:€€€€€€€€€

serving_default_itag4_originPlaceholder*'
_output_shapes
:€€€€€€€€€*
dtype0*
shape:€€€€€€€€€

serving_default_itag4_squarePlaceholder*'
_output_shapes
:€€€€€€€€€*
dtype0*
shape:€€€€€€€€€
v
serving_default_uidPlaceholder*'
_output_shapes
:€€€€€€€€€*
dtype0	*
shape:€€€€€€€€€
x
serving_default_utag1Placeholder*'
_output_shapes
:€€€€€€€€€*
dtype0	*
shape:€€€€€€€€€
x
serving_default_utag2Placeholder*'
_output_shapes
:€€€€€€€€€*
dtype0	*
shape:€€€€€€€€€
x
serving_default_utag3Placeholder*'
_output_shapes
:€€€€€€€€€*
dtype0	*
shape:€€€€€€€€€
x
serving_default_utag4Placeholder*'
_output_shapes
:€€€€€€€€€*
dtype0	*
shape:€€€€€€€€€
Ќ
StatefulPartitionedCallStatefulPartitionedCallserving_default_iidserving_default_itag1serving_default_itag2serving_default_itag3serving_default_itag4serving_default_itag4_cubeserving_default_itag4_originserving_default_itag4_squareserving_default_uidserving_default_utag1serving_default_utag2serving_default_utag3serving_default_utag4'fm_ranking_layer/embedding_1/embeddings%fm_ranking_layer/embedding/embeddingsfm_ranking_layer/bias*
Tin
2										*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *.
f)R'
%__inference_signature_wrapper_2189297
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
¬
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename)fm_ranking_layer/bias/Read/ReadVariableOp9fm_ranking_layer/embedding/embeddings/Read/ReadVariableOp;fm_ranking_layer/embedding_1/embeddings/Read/ReadVariableOpConst*
Tin	
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *)
f$R"
 __inference__traced_save_2189468
Б
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamefm_ranking_layer/bias%fm_ranking_layer/embedding/embeddings'fm_ranking_layer/embedding_1/embeddings*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *,
f'R%
#__inference__traced_restore_2189487Љњ
 
≠
2__inference_fm_ranking_layer_layer_call_fn_2189332

inputs_iid	
inputs_itag1	
inputs_itag2	
inputs_itag3	
inputs_itag4	
inputs_itag4_cube
inputs_itag4_origin
inputs_itag4_square

inputs_uid	
inputs_utag1	
inputs_utag2	
inputs_utag3	
inputs_utag4	
unknown:
рУ	
	unknown_0:
рУ	
	unknown_1:
identityИҐStatefulPartitionedCallЄ
StatefulPartitionedCallStatefulPartitionedCall
inputs_iidinputs_itag1inputs_itag2inputs_itag3inputs_itag4inputs_itag4_cubeinputs_itag4_origininputs_itag4_square
inputs_uidinputs_utag1inputs_utag2inputs_utag3inputs_utag4unknown	unknown_0	unknown_1*
Tin
2										*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *V
fQRO
M__inference_fm_ranking_layer_layer_call_and_return_conditional_losses_2188870o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Т
_input_shapesА
э:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
'
_output_shapes
:€€€€€€€€€
$
_user_specified_name
inputs/iid:UQ
'
_output_shapes
:€€€€€€€€€
&
_user_specified_nameinputs/itag1:UQ
'
_output_shapes
:€€€€€€€€€
&
_user_specified_nameinputs/itag2:UQ
'
_output_shapes
:€€€€€€€€€
&
_user_specified_nameinputs/itag3:UQ
'
_output_shapes
:€€€€€€€€€
&
_user_specified_nameinputs/itag4:ZV
'
_output_shapes
:€€€€€€€€€
+
_user_specified_nameinputs/itag4_cube:\X
'
_output_shapes
:€€€€€€€€€
-
_user_specified_nameinputs/itag4_origin:\X
'
_output_shapes
:€€€€€€€€€
-
_user_specified_nameinputs/itag4_square:SO
'
_output_shapes
:€€€€€€€€€
$
_user_specified_name
inputs/uid:U	Q
'
_output_shapes
:€€€€€€€€€
&
_user_specified_nameinputs/utag1:U
Q
'
_output_shapes
:€€€€€€€€€
&
_user_specified_nameinputs/utag2:UQ
'
_output_shapes
:€€€€€€€€€
&
_user_specified_nameinputs/utag3:UQ
'
_output_shapes
:€€€€€€€€€
&
_user_specified_nameinputs/utag4
ШB
’
M__inference_fm_ranking_layer_layer_call_and_return_conditional_losses_2189390

inputs_iid	
inputs_itag1	
inputs_itag2	
inputs_itag3	
inputs_itag4	
inputs_itag4_cube
inputs_itag4_origin
inputs_itag4_square

inputs_uid	
inputs_utag1	
inputs_utag2	
inputs_utag3	
inputs_utag4	8
$embedding_1_embedding_lookup_2189349:
рУ	6
"embedding_embedding_lookup_2189354:
рУ	%
readvariableop_resource:
identityИҐReadVariableOpҐembedding/embedding_lookupҐembedding_1/embedding_lookupҐGfm_ranking_layer/embedding/embeddings/Regularizer/Square/ReadVariableOpҐIfm_ranking_layer/embedding_1/embeddings/Regularizer/Square/ReadVariableOpM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :л
concatConcatV2
inputs_uid
inputs_iidinputs_utag1inputs_utag2inputs_utag3inputs_utag4inputs_itag1inputs_itag2inputs_itag3inputs_itag4concat/axis:output:0*
N
*
T0	*'
_output_shapes
:€€€€€€€€€
и
embedding_1/embedding_lookupResourceGather$embedding_1_embedding_lookup_2189349concat:output:0*
Tindices0	*7
_class-
+)loc:@embedding_1/embedding_lookup/2189349*+
_output_shapes
:€€€€€€€€€
*
dtype0«
%embedding_1/embedding_lookup/IdentityIdentity%embedding_1/embedding_lookup:output:0*
T0*7
_class-
+)loc:@embedding_1/embedding_lookup/2189349*+
_output_shapes
:€€€€€€€€€
Щ
'embedding_1/embedding_lookup/Identity_1Identity.embedding_1/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:€€€€€€€€€
в
embedding/embedding_lookupResourceGather"embedding_embedding_lookup_2189354concat:output:0*
Tindices0	*5
_class+
)'loc:@embedding/embedding_lookup/2189354*+
_output_shapes
:€€€€€€€€€
*
dtype0Ѕ
#embedding/embedding_lookup/IdentityIdentity#embedding/embedding_lookup:output:0*
T0*5
_class+
)'loc:@embedding/embedding_lookup/2189354*+
_output_shapes
:€€€€€€€€€
Х
%embedding/embedding_lookup/Identity_1Identity,embedding/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:€€€€€€€€€
W
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :О
SumSum0embedding_1/embedding_lookup/Identity_1:output:0Sum/reduction_indices:output:0*
T0*'
_output_shapes
:€€€€€€€€€v
SquareSquare.embedding/embedding_lookup/Identity_1:output:0*
T0*+
_output_shapes
:€€€€€€€€€
Y
Sum_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :l
Sum_1Sum
Square:y:0 Sum_1/reduction_indices:output:0*
T0*'
_output_shapes
:€€€€€€€€€Y
Sum_2/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :Р
Sum_2Sum.embedding/embedding_lookup/Identity_1:output:0 Sum_2/reduction_indices:output:0*
T0*'
_output_shapes
:€€€€€€€€€T
Square_1SquareSum_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€Z
SubSubSquare_1:y:0Sum_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€Y
Sum_3/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :z
Sum_3SumSub:z:0 Sum_3/reduction_indices:output:0*
T0*'
_output_shapes
:€€€€€€€€€*
	keep_dims(J
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?\
mulMulmul/x:output:0Sum_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0d
addAddV2ReadVariableOp:value:0Sum:output:0*
T0*'
_output_shapes
:€€€€€€€€€R
add_1AddV2add:z:0mul:z:0*
T0*'
_output_shapes
:€€€€€€€€€O
SigmoidSigmoid	add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ђ
Gfm_ranking_layer/embedding/embeddings/Regularizer/Square/ReadVariableOpReadVariableOp"embedding_embedding_lookup_2189354* 
_output_shapes
:
рУ	*
dtype0Њ
8fm_ranking_layer/embedding/embeddings/Regularizer/SquareSquareOfm_ranking_layer/embedding/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
рУ	И
7fm_ranking_layer/embedding/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       Ё
5fm_ranking_layer/embedding/embeddings/Regularizer/SumSum<fm_ranking_layer/embedding/embeddings/Regularizer/Square:y:0@fm_ranking_layer/embedding/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: |
7fm_ranking_layer/embedding/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<я
5fm_ranking_layer/embedding/embeddings/Regularizer/mulMul@fm_ranking_layer/embedding/embeddings/Regularizer/mul/x:output:0>fm_ranking_layer/embedding/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: ∞
Ifm_ranking_layer/embedding_1/embeddings/Regularizer/Square/ReadVariableOpReadVariableOp$embedding_1_embedding_lookup_2189349* 
_output_shapes
:
рУ	*
dtype0¬
:fm_ranking_layer/embedding_1/embeddings/Regularizer/SquareSquareQfm_ranking_layer/embedding_1/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
рУ	К
9fm_ranking_layer/embedding_1/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       г
7fm_ranking_layer/embedding_1/embeddings/Regularizer/SumSum>fm_ranking_layer/embedding_1/embeddings/Regularizer/Square:y:0Bfm_ranking_layer/embedding_1/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: ~
9fm_ranking_layer/embedding_1/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<е
7fm_ranking_layer/embedding_1/embeddings/Regularizer/mulMulBfm_ranking_layer/embedding_1/embeddings/Regularizer/mul/x:output:0@fm_ranking_layer/embedding_1/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€©
NoOpNoOp^ReadVariableOp^embedding/embedding_lookup^embedding_1/embedding_lookupH^fm_ranking_layer/embedding/embeddings/Regularizer/Square/ReadVariableOpJ^fm_ranking_layer/embedding_1/embeddings/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Т
_input_shapesА
э:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€: : : 2 
ReadVariableOpReadVariableOp28
embedding/embedding_lookupembedding/embedding_lookup2<
embedding_1/embedding_lookupembedding_1/embedding_lookup2Т
Gfm_ranking_layer/embedding/embeddings/Regularizer/Square/ReadVariableOpGfm_ranking_layer/embedding/embeddings/Regularizer/Square/ReadVariableOp2Ц
Ifm_ranking_layer/embedding_1/embeddings/Regularizer/Square/ReadVariableOpIfm_ranking_layer/embedding_1/embeddings/Regularizer/Square/ReadVariableOp:S O
'
_output_shapes
:€€€€€€€€€
$
_user_specified_name
inputs/iid:UQ
'
_output_shapes
:€€€€€€€€€
&
_user_specified_nameinputs/itag1:UQ
'
_output_shapes
:€€€€€€€€€
&
_user_specified_nameinputs/itag2:UQ
'
_output_shapes
:€€€€€€€€€
&
_user_specified_nameinputs/itag3:UQ
'
_output_shapes
:€€€€€€€€€
&
_user_specified_nameinputs/itag4:ZV
'
_output_shapes
:€€€€€€€€€
+
_user_specified_nameinputs/itag4_cube:\X
'
_output_shapes
:€€€€€€€€€
-
_user_specified_nameinputs/itag4_origin:\X
'
_output_shapes
:€€€€€€€€€
-
_user_specified_nameinputs/itag4_square:SO
'
_output_shapes
:€€€€€€€€€
$
_user_specified_name
inputs/uid:U	Q
'
_output_shapes
:€€€€€€€€€
&
_user_specified_nameinputs/utag1:U
Q
'
_output_shapes
:€€€€€€€€€
&
_user_specified_nameinputs/utag2:UQ
'
_output_shapes
:€€€€€€€€€
&
_user_specified_nameinputs/utag3:UQ
'
_output_shapes
:€€€€€€€€€
&
_user_specified_nameinputs/utag4
С+
—
B__inference_model_layer_call_and_return_conditional_losses_2188891

inputs	
inputs_1	
inputs_2	
inputs_3	
inputs_4	
inputs_5
inputs_6
inputs_7
inputs_8	
inputs_9	
	inputs_10	
	inputs_11	
	inputs_12	,
fm_ranking_layer_2188871:
рУ	,
fm_ranking_layer_2188873:
рУ	&
fm_ranking_layer_2188875:
identityИҐ(fm_ranking_layer/StatefulPartitionedCallҐGfm_ranking_layer/embedding/embeddings/Regularizer/Square/ReadVariableOpҐIfm_ranking_layer/embedding_1/embeddings/Regularizer/Square/ReadVariableOpґ
(fm_ranking_layer/StatefulPartitionedCallStatefulPartitionedCallinputsinputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7inputs_8inputs_9	inputs_10	inputs_11	inputs_12fm_ranking_layer_2188871fm_ranking_layer_2188873fm_ranking_layer_2188875*
Tin
2										*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *V
fQRO
M__inference_fm_ranking_layer_layer_call_and_return_conditional_losses_2188870Ґ
Gfm_ranking_layer/embedding/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpfm_ranking_layer_2188873* 
_output_shapes
:
рУ	*
dtype0Њ
8fm_ranking_layer/embedding/embeddings/Regularizer/SquareSquareOfm_ranking_layer/embedding/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
рУ	И
7fm_ranking_layer/embedding/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       Ё
5fm_ranking_layer/embedding/embeddings/Regularizer/SumSum<fm_ranking_layer/embedding/embeddings/Regularizer/Square:y:0@fm_ranking_layer/embedding/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: |
7fm_ranking_layer/embedding/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<я
5fm_ranking_layer/embedding/embeddings/Regularizer/mulMul@fm_ranking_layer/embedding/embeddings/Regularizer/mul/x:output:0>fm_ranking_layer/embedding/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: §
Ifm_ranking_layer/embedding_1/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpfm_ranking_layer_2188871* 
_output_shapes
:
рУ	*
dtype0¬
:fm_ranking_layer/embedding_1/embeddings/Regularizer/SquareSquareQfm_ranking_layer/embedding_1/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
рУ	К
9fm_ranking_layer/embedding_1/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       г
7fm_ranking_layer/embedding_1/embeddings/Regularizer/SumSum>fm_ranking_layer/embedding_1/embeddings/Regularizer/Square:y:0Bfm_ranking_layer/embedding_1/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: ~
9fm_ranking_layer/embedding_1/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<е
7fm_ranking_layer/embedding_1/embeddings/Regularizer/mulMulBfm_ranking_layer/embedding_1/embeddings/Regularizer/mul/x:output:0@fm_ranking_layer/embedding_1/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: А
IdentityIdentity1fm_ranking_layer/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€З
NoOpNoOp)^fm_ranking_layer/StatefulPartitionedCallH^fm_ranking_layer/embedding/embeddings/Regularizer/Square/ReadVariableOpJ^fm_ranking_layer/embedding_1/embeddings/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Т
_input_shapesА
э:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€: : : 2T
(fm_ranking_layer/StatefulPartitionedCall(fm_ranking_layer/StatefulPartitionedCall2Т
Gfm_ranking_layer/embedding/embeddings/Regularizer/Square/ReadVariableOpGfm_ranking_layer/embedding/embeddings/Regularizer/Square/ReadVariableOp2Ц
Ifm_ranking_layer/embedding_1/embeddings/Regularizer/Square/ReadVariableOpIfm_ranking_layer/embedding_1/embeddings/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:O	K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:O
K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
°L
∞
B__inference_model_layer_call_and_return_conditional_losses_2189214

inputs_iid	
inputs_itag1	
inputs_itag2	
inputs_itag3	
inputs_itag4	
inputs_itag4_cube
inputs_itag4_origin
inputs_itag4_square

inputs_uid	
inputs_utag1	
inputs_utag2	
inputs_utag3	
inputs_utag4	I
5fm_ranking_layer_embedding_1_embedding_lookup_2189173:
рУ	G
3fm_ranking_layer_embedding_embedding_lookup_2189178:
рУ	6
(fm_ranking_layer_readvariableop_resource:
identityИҐfm_ranking_layer/ReadVariableOpҐ+fm_ranking_layer/embedding/embedding_lookupҐGfm_ranking_layer/embedding/embeddings/Regularizer/Square/ReadVariableOpҐ-fm_ranking_layer/embedding_1/embedding_lookupҐIfm_ranking_layer/embedding_1/embeddings/Regularizer/Square/ReadVariableOp^
fm_ranking_layer/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Н
fm_ranking_layer/concatConcatV2
inputs_uid
inputs_iidinputs_utag1inputs_utag2inputs_utag3inputs_utag4inputs_itag1inputs_itag2inputs_itag3inputs_itag4%fm_ranking_layer/concat/axis:output:0*
N
*
T0	*'
_output_shapes
:€€€€€€€€€
ђ
-fm_ranking_layer/embedding_1/embedding_lookupResourceGather5fm_ranking_layer_embedding_1_embedding_lookup_2189173 fm_ranking_layer/concat:output:0*
Tindices0	*H
_class>
<:loc:@fm_ranking_layer/embedding_1/embedding_lookup/2189173*+
_output_shapes
:€€€€€€€€€
*
dtype0ъ
6fm_ranking_layer/embedding_1/embedding_lookup/IdentityIdentity6fm_ranking_layer/embedding_1/embedding_lookup:output:0*
T0*H
_class>
<:loc:@fm_ranking_layer/embedding_1/embedding_lookup/2189173*+
_output_shapes
:€€€€€€€€€
ї
8fm_ranking_layer/embedding_1/embedding_lookup/Identity_1Identity?fm_ranking_layer/embedding_1/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:€€€€€€€€€
¶
+fm_ranking_layer/embedding/embedding_lookupResourceGather3fm_ranking_layer_embedding_embedding_lookup_2189178 fm_ranking_layer/concat:output:0*
Tindices0	*F
_class<
:8loc:@fm_ranking_layer/embedding/embedding_lookup/2189178*+
_output_shapes
:€€€€€€€€€
*
dtype0ф
4fm_ranking_layer/embedding/embedding_lookup/IdentityIdentity4fm_ranking_layer/embedding/embedding_lookup:output:0*
T0*F
_class<
:8loc:@fm_ranking_layer/embedding/embedding_lookup/2189178*+
_output_shapes
:€€€€€€€€€
Ј
6fm_ranking_layer/embedding/embedding_lookup/Identity_1Identity=fm_ranking_layer/embedding/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:€€€€€€€€€
h
&fm_ranking_layer/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :Ѕ
fm_ranking_layer/SumSumAfm_ranking_layer/embedding_1/embedding_lookup/Identity_1:output:0/fm_ranking_layer/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:€€€€€€€€€Ш
fm_ranking_layer/SquareSquare?fm_ranking_layer/embedding/embedding_lookup/Identity_1:output:0*
T0*+
_output_shapes
:€€€€€€€€€
j
(fm_ranking_layer/Sum_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :Я
fm_ranking_layer/Sum_1Sumfm_ranking_layer/Square:y:01fm_ranking_layer/Sum_1/reduction_indices:output:0*
T0*'
_output_shapes
:€€€€€€€€€j
(fm_ranking_layer/Sum_2/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :√
fm_ranking_layer/Sum_2Sum?fm_ranking_layer/embedding/embedding_lookup/Identity_1:output:01fm_ranking_layer/Sum_2/reduction_indices:output:0*
T0*'
_output_shapes
:€€€€€€€€€v
fm_ranking_layer/Square_1Squarefm_ranking_layer/Sum_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€Н
fm_ranking_layer/SubSubfm_ranking_layer/Square_1:y:0fm_ranking_layer/Sum_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€j
(fm_ranking_layer/Sum_3/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :≠
fm_ranking_layer/Sum_3Sumfm_ranking_layer/Sub:z:01fm_ranking_layer/Sum_3/reduction_indices:output:0*
T0*'
_output_shapes
:€€€€€€€€€*
	keep_dims([
fm_ranking_layer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?П
fm_ranking_layer/mulMulfm_ranking_layer/mul/x:output:0fm_ranking_layer/Sum_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€Д
fm_ranking_layer/ReadVariableOpReadVariableOp(fm_ranking_layer_readvariableop_resource*
_output_shapes
:*
dtype0Ч
fm_ranking_layer/addAddV2'fm_ranking_layer/ReadVariableOp:value:0fm_ranking_layer/Sum:output:0*
T0*'
_output_shapes
:€€€€€€€€€Е
fm_ranking_layer/add_1AddV2fm_ranking_layer/add:z:0fm_ranking_layer/mul:z:0*
T0*'
_output_shapes
:€€€€€€€€€q
fm_ranking_layer/SigmoidSigmoidfm_ranking_layer/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€љ
Gfm_ranking_layer/embedding/embeddings/Regularizer/Square/ReadVariableOpReadVariableOp3fm_ranking_layer_embedding_embedding_lookup_2189178* 
_output_shapes
:
рУ	*
dtype0Њ
8fm_ranking_layer/embedding/embeddings/Regularizer/SquareSquareOfm_ranking_layer/embedding/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
рУ	И
7fm_ranking_layer/embedding/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       Ё
5fm_ranking_layer/embedding/embeddings/Regularizer/SumSum<fm_ranking_layer/embedding/embeddings/Regularizer/Square:y:0@fm_ranking_layer/embedding/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: |
7fm_ranking_layer/embedding/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<я
5fm_ranking_layer/embedding/embeddings/Regularizer/mulMul@fm_ranking_layer/embedding/embeddings/Regularizer/mul/x:output:0>fm_ranking_layer/embedding/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: Ѕ
Ifm_ranking_layer/embedding_1/embeddings/Regularizer/Square/ReadVariableOpReadVariableOp5fm_ranking_layer_embedding_1_embedding_lookup_2189173* 
_output_shapes
:
рУ	*
dtype0¬
:fm_ranking_layer/embedding_1/embeddings/Regularizer/SquareSquareQfm_ranking_layer/embedding_1/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
рУ	К
9fm_ranking_layer/embedding_1/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       г
7fm_ranking_layer/embedding_1/embeddings/Regularizer/SumSum>fm_ranking_layer/embedding_1/embeddings/Regularizer/Square:y:0Bfm_ranking_layer/embedding_1/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: ~
9fm_ranking_layer/embedding_1/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<е
7fm_ranking_layer/embedding_1/embeddings/Regularizer/mulMulBfm_ranking_layer/embedding_1/embeddings/Regularizer/mul/x:output:0@fm_ranking_layer/embedding_1/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: k
IdentityIdentityfm_ranking_layer/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€№
NoOpNoOp ^fm_ranking_layer/ReadVariableOp,^fm_ranking_layer/embedding/embedding_lookupH^fm_ranking_layer/embedding/embeddings/Regularizer/Square/ReadVariableOp.^fm_ranking_layer/embedding_1/embedding_lookupJ^fm_ranking_layer/embedding_1/embeddings/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Т
_input_shapesА
э:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€: : : 2B
fm_ranking_layer/ReadVariableOpfm_ranking_layer/ReadVariableOp2Z
+fm_ranking_layer/embedding/embedding_lookup+fm_ranking_layer/embedding/embedding_lookup2Т
Gfm_ranking_layer/embedding/embeddings/Regularizer/Square/ReadVariableOpGfm_ranking_layer/embedding/embeddings/Regularizer/Square/ReadVariableOp2^
-fm_ranking_layer/embedding_1/embedding_lookup-fm_ranking_layer/embedding_1/embedding_lookup2Ц
Ifm_ranking_layer/embedding_1/embeddings/Regularizer/Square/ReadVariableOpIfm_ranking_layer/embedding_1/embeddings/Regularizer/Square/ReadVariableOp:S O
'
_output_shapes
:€€€€€€€€€
$
_user_specified_name
inputs/iid:UQ
'
_output_shapes
:€€€€€€€€€
&
_user_specified_nameinputs/itag1:UQ
'
_output_shapes
:€€€€€€€€€
&
_user_specified_nameinputs/itag2:UQ
'
_output_shapes
:€€€€€€€€€
&
_user_specified_nameinputs/itag3:UQ
'
_output_shapes
:€€€€€€€€€
&
_user_specified_nameinputs/itag4:ZV
'
_output_shapes
:€€€€€€€€€
+
_user_specified_nameinputs/itag4_cube:\X
'
_output_shapes
:€€€€€€€€€
-
_user_specified_nameinputs/itag4_origin:\X
'
_output_shapes
:€€€€€€€€€
-
_user_specified_nameinputs/itag4_square:SO
'
_output_shapes
:€€€€€€€€€
$
_user_specified_name
inputs/uid:U	Q
'
_output_shapes
:€€€€€€€€€
&
_user_specified_nameinputs/utag1:U
Q
'
_output_shapes
:€€€€€€€€€
&
_user_specified_nameinputs/utag2:UQ
'
_output_shapes
:€€€€€€€€€
&
_user_specified_nameinputs/utag3:UQ
'
_output_shapes
:€€€€€€€€€
&
_user_specified_nameinputs/utag4
Ќ
а
__inference_loss_fn_0_2189413d
Pfm_ranking_layer_embedding_embeddings_regularizer_square_readvariableop_resource:
рУ	
identityИҐGfm_ranking_layer/embedding/embeddings/Regularizer/Square/ReadVariableOpЏ
Gfm_ranking_layer/embedding/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpPfm_ranking_layer_embedding_embeddings_regularizer_square_readvariableop_resource* 
_output_shapes
:
рУ	*
dtype0Њ
8fm_ranking_layer/embedding/embeddings/Regularizer/SquareSquareOfm_ranking_layer/embedding/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
рУ	И
7fm_ranking_layer/embedding/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       Ё
5fm_ranking_layer/embedding/embeddings/Regularizer/SumSum<fm_ranking_layer/embedding/embeddings/Regularizer/Square:y:0@fm_ranking_layer/embedding/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: |
7fm_ranking_layer/embedding/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<я
5fm_ranking_layer/embedding/embeddings/Regularizer/mulMul@fm_ranking_layer/embedding/embeddings/Regularizer/mul/x:output:0>fm_ranking_layer/embedding/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: w
IdentityIdentity9fm_ranking_layer/embedding/embeddings/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: Р
NoOpNoOpH^fm_ranking_layer/embedding/embeddings/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2Т
Gfm_ranking_layer/embedding/embeddings/Regularizer/Square/ReadVariableOpGfm_ranking_layer/embedding/embeddings/Regularizer/Square/ReadVariableOp
£
«
'__inference_model_layer_call_fn_2188900
iid		
itag1		
itag2		
itag3		
itag4	

itag4_cube
itag4_origin
itag4_square
uid		
utag1		
utag2		
utag3		
utag4	
unknown:
рУ	
	unknown_0:
рУ	
	unknown_1:
identityИҐStatefulPartitionedCall“
StatefulPartitionedCallStatefulPartitionedCalliiditag1itag2itag3itag4
itag4_cubeitag4_originitag4_squareuidutag1utag2utag3utag4unknown	unknown_0	unknown_1*
Tin
2										*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_model_layer_call_and_return_conditional_losses_2188891o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Т
_input_shapesА
э:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€: : : 22
StatefulPartitionedCallStatefulPartitionedCall:L H
'
_output_shapes
:€€€€€€€€€

_user_specified_nameiid:NJ
'
_output_shapes
:€€€€€€€€€

_user_specified_nameitag1:NJ
'
_output_shapes
:€€€€€€€€€

_user_specified_nameitag2:NJ
'
_output_shapes
:€€€€€€€€€

_user_specified_nameitag3:NJ
'
_output_shapes
:€€€€€€€€€

_user_specified_nameitag4:SO
'
_output_shapes
:€€€€€€€€€
$
_user_specified_name
itag4_cube:UQ
'
_output_shapes
:€€€€€€€€€
&
_user_specified_nameitag4_origin:UQ
'
_output_shapes
:€€€€€€€€€
&
_user_specified_nameitag4_square:LH
'
_output_shapes
:€€€€€€€€€

_user_specified_nameuid:N	J
'
_output_shapes
:€€€€€€€€€

_user_specified_nameutag1:N
J
'
_output_shapes
:€€€€€€€€€

_user_specified_nameutag2:NJ
'
_output_shapes
:€€€€€€€€€

_user_specified_nameutag3:NJ
'
_output_shapes
:€€€€€€€€€

_user_specified_nameutag4
і
Ґ
'__inference_model_layer_call_fn_2189156

inputs_iid	
inputs_itag1	
inputs_itag2	
inputs_itag3	
inputs_itag4	
inputs_itag4_cube
inputs_itag4_origin
inputs_itag4_square

inputs_uid	
inputs_utag1	
inputs_utag2	
inputs_utag3	
inputs_utag4	
unknown:
рУ	
	unknown_0:
рУ	
	unknown_1:
identityИҐStatefulPartitionedCall≠
StatefulPartitionedCallStatefulPartitionedCall
inputs_iidinputs_itag1inputs_itag2inputs_itag3inputs_itag4inputs_itag4_cubeinputs_itag4_origininputs_itag4_square
inputs_uidinputs_utag1inputs_utag2inputs_utag3inputs_utag4unknown	unknown_0	unknown_1*
Tin
2										*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_model_layer_call_and_return_conditional_losses_2188996o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Т
_input_shapesА
э:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
'
_output_shapes
:€€€€€€€€€
$
_user_specified_name
inputs/iid:UQ
'
_output_shapes
:€€€€€€€€€
&
_user_specified_nameinputs/itag1:UQ
'
_output_shapes
:€€€€€€€€€
&
_user_specified_nameinputs/itag2:UQ
'
_output_shapes
:€€€€€€€€€
&
_user_specified_nameinputs/itag3:UQ
'
_output_shapes
:€€€€€€€€€
&
_user_specified_nameinputs/itag4:ZV
'
_output_shapes
:€€€€€€€€€
+
_user_specified_nameinputs/itag4_cube:\X
'
_output_shapes
:€€€€€€€€€
-
_user_specified_nameinputs/itag4_origin:\X
'
_output_shapes
:€€€€€€€€€
-
_user_specified_nameinputs/itag4_square:SO
'
_output_shapes
:€€€€€€€€€
$
_user_specified_name
inputs/uid:U	Q
'
_output_shapes
:€€€€€€€€€
&
_user_specified_nameinputs/utag1:U
Q
'
_output_shapes
:€€€€€€€€€
&
_user_specified_nameinputs/utag2:UQ
'
_output_shapes
:€€€€€€€€€
&
_user_specified_nameinputs/utag3:UQ
'
_output_shapes
:€€€€€€€€€
&
_user_specified_nameinputs/utag4
с
д
__inference_loss_fn_1_2189424f
Rfm_ranking_layer_embedding_1_embeddings_regularizer_square_readvariableop_resource:
рУ	
identityИҐIfm_ranking_layer/embedding_1/embeddings/Regularizer/Square/ReadVariableOpё
Ifm_ranking_layer/embedding_1/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpRfm_ranking_layer_embedding_1_embeddings_regularizer_square_readvariableop_resource* 
_output_shapes
:
рУ	*
dtype0¬
:fm_ranking_layer/embedding_1/embeddings/Regularizer/SquareSquareQfm_ranking_layer/embedding_1/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
рУ	К
9fm_ranking_layer/embedding_1/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       г
7fm_ranking_layer/embedding_1/embeddings/Regularizer/SumSum>fm_ranking_layer/embedding_1/embeddings/Regularizer/Square:y:0Bfm_ranking_layer/embedding_1/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: ~
9fm_ranking_layer/embedding_1/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<е
7fm_ranking_layer/embedding_1/embeddings/Regularizer/mulMulBfm_ranking_layer/embedding_1/embeddings/Regularizer/mul/x:output:0@fm_ranking_layer/embedding_1/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: y
IdentityIdentity;fm_ranking_layer/embedding_1/embeddings/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: Т
NoOpNoOpJ^fm_ranking_layer/embedding_1/embeddings/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2Ц
Ifm_ranking_layer/embedding_1/embeddings/Regularizer/Square/ReadVariableOpIfm_ranking_layer/embedding_1/embeddings/Regularizer/Square/ReadVariableOp
С+
—
B__inference_model_layer_call_and_return_conditional_losses_2188996

inputs	
inputs_1	
inputs_2	
inputs_3	
inputs_4	
inputs_5
inputs_6
inputs_7
inputs_8	
inputs_9	
	inputs_10	
	inputs_11	
	inputs_12	,
fm_ranking_layer_2188976:
рУ	,
fm_ranking_layer_2188978:
рУ	&
fm_ranking_layer_2188980:
identityИҐ(fm_ranking_layer/StatefulPartitionedCallҐGfm_ranking_layer/embedding/embeddings/Regularizer/Square/ReadVariableOpҐIfm_ranking_layer/embedding_1/embeddings/Regularizer/Square/ReadVariableOpґ
(fm_ranking_layer/StatefulPartitionedCallStatefulPartitionedCallinputsinputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7inputs_8inputs_9	inputs_10	inputs_11	inputs_12fm_ranking_layer_2188976fm_ranking_layer_2188978fm_ranking_layer_2188980*
Tin
2										*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *V
fQRO
M__inference_fm_ranking_layer_layer_call_and_return_conditional_losses_2188870Ґ
Gfm_ranking_layer/embedding/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpfm_ranking_layer_2188978* 
_output_shapes
:
рУ	*
dtype0Њ
8fm_ranking_layer/embedding/embeddings/Regularizer/SquareSquareOfm_ranking_layer/embedding/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
рУ	И
7fm_ranking_layer/embedding/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       Ё
5fm_ranking_layer/embedding/embeddings/Regularizer/SumSum<fm_ranking_layer/embedding/embeddings/Regularizer/Square:y:0@fm_ranking_layer/embedding/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: |
7fm_ranking_layer/embedding/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<я
5fm_ranking_layer/embedding/embeddings/Regularizer/mulMul@fm_ranking_layer/embedding/embeddings/Regularizer/mul/x:output:0>fm_ranking_layer/embedding/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: §
Ifm_ranking_layer/embedding_1/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpfm_ranking_layer_2188976* 
_output_shapes
:
рУ	*
dtype0¬
:fm_ranking_layer/embedding_1/embeddings/Regularizer/SquareSquareQfm_ranking_layer/embedding_1/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
рУ	К
9fm_ranking_layer/embedding_1/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       г
7fm_ranking_layer/embedding_1/embeddings/Regularizer/SumSum>fm_ranking_layer/embedding_1/embeddings/Regularizer/Square:y:0Bfm_ranking_layer/embedding_1/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: ~
9fm_ranking_layer/embedding_1/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<е
7fm_ranking_layer/embedding_1/embeddings/Regularizer/mulMulBfm_ranking_layer/embedding_1/embeddings/Regularizer/mul/x:output:0@fm_ranking_layer/embedding_1/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: А
IdentityIdentity1fm_ranking_layer/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€З
NoOpNoOp)^fm_ranking_layer/StatefulPartitionedCallH^fm_ranking_layer/embedding/embeddings/Regularizer/Square/ReadVariableOpJ^fm_ranking_layer/embedding_1/embeddings/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Т
_input_shapesА
э:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€: : : 2T
(fm_ranking_layer/StatefulPartitionedCall(fm_ranking_layer/StatefulPartitionedCall2Т
Gfm_ranking_layer/embedding/embeddings/Regularizer/Square/ReadVariableOpGfm_ranking_layer/embedding/embeddings/Regularizer/Square/ReadVariableOp2Ц
Ifm_ranking_layer/embedding_1/embeddings/Regularizer/Square/ReadVariableOpIfm_ranking_layer/embedding_1/embeddings/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:O	K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:O
K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
°L
∞
B__inference_model_layer_call_and_return_conditional_losses_2189272

inputs_iid	
inputs_itag1	
inputs_itag2	
inputs_itag3	
inputs_itag4	
inputs_itag4_cube
inputs_itag4_origin
inputs_itag4_square

inputs_uid	
inputs_utag1	
inputs_utag2	
inputs_utag3	
inputs_utag4	I
5fm_ranking_layer_embedding_1_embedding_lookup_2189231:
рУ	G
3fm_ranking_layer_embedding_embedding_lookup_2189236:
рУ	6
(fm_ranking_layer_readvariableop_resource:
identityИҐfm_ranking_layer/ReadVariableOpҐ+fm_ranking_layer/embedding/embedding_lookupҐGfm_ranking_layer/embedding/embeddings/Regularizer/Square/ReadVariableOpҐ-fm_ranking_layer/embedding_1/embedding_lookupҐIfm_ranking_layer/embedding_1/embeddings/Regularizer/Square/ReadVariableOp^
fm_ranking_layer/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Н
fm_ranking_layer/concatConcatV2
inputs_uid
inputs_iidinputs_utag1inputs_utag2inputs_utag3inputs_utag4inputs_itag1inputs_itag2inputs_itag3inputs_itag4%fm_ranking_layer/concat/axis:output:0*
N
*
T0	*'
_output_shapes
:€€€€€€€€€
ђ
-fm_ranking_layer/embedding_1/embedding_lookupResourceGather5fm_ranking_layer_embedding_1_embedding_lookup_2189231 fm_ranking_layer/concat:output:0*
Tindices0	*H
_class>
<:loc:@fm_ranking_layer/embedding_1/embedding_lookup/2189231*+
_output_shapes
:€€€€€€€€€
*
dtype0ъ
6fm_ranking_layer/embedding_1/embedding_lookup/IdentityIdentity6fm_ranking_layer/embedding_1/embedding_lookup:output:0*
T0*H
_class>
<:loc:@fm_ranking_layer/embedding_1/embedding_lookup/2189231*+
_output_shapes
:€€€€€€€€€
ї
8fm_ranking_layer/embedding_1/embedding_lookup/Identity_1Identity?fm_ranking_layer/embedding_1/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:€€€€€€€€€
¶
+fm_ranking_layer/embedding/embedding_lookupResourceGather3fm_ranking_layer_embedding_embedding_lookup_2189236 fm_ranking_layer/concat:output:0*
Tindices0	*F
_class<
:8loc:@fm_ranking_layer/embedding/embedding_lookup/2189236*+
_output_shapes
:€€€€€€€€€
*
dtype0ф
4fm_ranking_layer/embedding/embedding_lookup/IdentityIdentity4fm_ranking_layer/embedding/embedding_lookup:output:0*
T0*F
_class<
:8loc:@fm_ranking_layer/embedding/embedding_lookup/2189236*+
_output_shapes
:€€€€€€€€€
Ј
6fm_ranking_layer/embedding/embedding_lookup/Identity_1Identity=fm_ranking_layer/embedding/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:€€€€€€€€€
h
&fm_ranking_layer/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :Ѕ
fm_ranking_layer/SumSumAfm_ranking_layer/embedding_1/embedding_lookup/Identity_1:output:0/fm_ranking_layer/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:€€€€€€€€€Ш
fm_ranking_layer/SquareSquare?fm_ranking_layer/embedding/embedding_lookup/Identity_1:output:0*
T0*+
_output_shapes
:€€€€€€€€€
j
(fm_ranking_layer/Sum_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :Я
fm_ranking_layer/Sum_1Sumfm_ranking_layer/Square:y:01fm_ranking_layer/Sum_1/reduction_indices:output:0*
T0*'
_output_shapes
:€€€€€€€€€j
(fm_ranking_layer/Sum_2/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :√
fm_ranking_layer/Sum_2Sum?fm_ranking_layer/embedding/embedding_lookup/Identity_1:output:01fm_ranking_layer/Sum_2/reduction_indices:output:0*
T0*'
_output_shapes
:€€€€€€€€€v
fm_ranking_layer/Square_1Squarefm_ranking_layer/Sum_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€Н
fm_ranking_layer/SubSubfm_ranking_layer/Square_1:y:0fm_ranking_layer/Sum_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€j
(fm_ranking_layer/Sum_3/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :≠
fm_ranking_layer/Sum_3Sumfm_ranking_layer/Sub:z:01fm_ranking_layer/Sum_3/reduction_indices:output:0*
T0*'
_output_shapes
:€€€€€€€€€*
	keep_dims([
fm_ranking_layer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?П
fm_ranking_layer/mulMulfm_ranking_layer/mul/x:output:0fm_ranking_layer/Sum_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€Д
fm_ranking_layer/ReadVariableOpReadVariableOp(fm_ranking_layer_readvariableop_resource*
_output_shapes
:*
dtype0Ч
fm_ranking_layer/addAddV2'fm_ranking_layer/ReadVariableOp:value:0fm_ranking_layer/Sum:output:0*
T0*'
_output_shapes
:€€€€€€€€€Е
fm_ranking_layer/add_1AddV2fm_ranking_layer/add:z:0fm_ranking_layer/mul:z:0*
T0*'
_output_shapes
:€€€€€€€€€q
fm_ranking_layer/SigmoidSigmoidfm_ranking_layer/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€љ
Gfm_ranking_layer/embedding/embeddings/Regularizer/Square/ReadVariableOpReadVariableOp3fm_ranking_layer_embedding_embedding_lookup_2189236* 
_output_shapes
:
рУ	*
dtype0Њ
8fm_ranking_layer/embedding/embeddings/Regularizer/SquareSquareOfm_ranking_layer/embedding/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
рУ	И
7fm_ranking_layer/embedding/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       Ё
5fm_ranking_layer/embedding/embeddings/Regularizer/SumSum<fm_ranking_layer/embedding/embeddings/Regularizer/Square:y:0@fm_ranking_layer/embedding/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: |
7fm_ranking_layer/embedding/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<я
5fm_ranking_layer/embedding/embeddings/Regularizer/mulMul@fm_ranking_layer/embedding/embeddings/Regularizer/mul/x:output:0>fm_ranking_layer/embedding/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: Ѕ
Ifm_ranking_layer/embedding_1/embeddings/Regularizer/Square/ReadVariableOpReadVariableOp5fm_ranking_layer_embedding_1_embedding_lookup_2189231* 
_output_shapes
:
рУ	*
dtype0¬
:fm_ranking_layer/embedding_1/embeddings/Regularizer/SquareSquareQfm_ranking_layer/embedding_1/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
рУ	К
9fm_ranking_layer/embedding_1/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       г
7fm_ranking_layer/embedding_1/embeddings/Regularizer/SumSum>fm_ranking_layer/embedding_1/embeddings/Regularizer/Square:y:0Bfm_ranking_layer/embedding_1/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: ~
9fm_ranking_layer/embedding_1/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<е
7fm_ranking_layer/embedding_1/embeddings/Regularizer/mulMulBfm_ranking_layer/embedding_1/embeddings/Regularizer/mul/x:output:0@fm_ranking_layer/embedding_1/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: k
IdentityIdentityfm_ranking_layer/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€№
NoOpNoOp ^fm_ranking_layer/ReadVariableOp,^fm_ranking_layer/embedding/embedding_lookupH^fm_ranking_layer/embedding/embeddings/Regularizer/Square/ReadVariableOp.^fm_ranking_layer/embedding_1/embedding_lookupJ^fm_ranking_layer/embedding_1/embeddings/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Т
_input_shapesА
э:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€: : : 2B
fm_ranking_layer/ReadVariableOpfm_ranking_layer/ReadVariableOp2Z
+fm_ranking_layer/embedding/embedding_lookup+fm_ranking_layer/embedding/embedding_lookup2Т
Gfm_ranking_layer/embedding/embeddings/Regularizer/Square/ReadVariableOpGfm_ranking_layer/embedding/embeddings/Regularizer/Square/ReadVariableOp2^
-fm_ranking_layer/embedding_1/embedding_lookup-fm_ranking_layer/embedding_1/embedding_lookup2Ц
Ifm_ranking_layer/embedding_1/embeddings/Regularizer/Square/ReadVariableOpIfm_ranking_layer/embedding_1/embeddings/Regularizer/Square/ReadVariableOp:S O
'
_output_shapes
:€€€€€€€€€
$
_user_specified_name
inputs/iid:UQ
'
_output_shapes
:€€€€€€€€€
&
_user_specified_nameinputs/itag1:UQ
'
_output_shapes
:€€€€€€€€€
&
_user_specified_nameinputs/itag2:UQ
'
_output_shapes
:€€€€€€€€€
&
_user_specified_nameinputs/itag3:UQ
'
_output_shapes
:€€€€€€€€€
&
_user_specified_nameinputs/itag4:ZV
'
_output_shapes
:€€€€€€€€€
+
_user_specified_nameinputs/itag4_cube:\X
'
_output_shapes
:€€€€€€€€€
-
_user_specified_nameinputs/itag4_origin:\X
'
_output_shapes
:€€€€€€€€€
-
_user_specified_nameinputs/itag4_square:SO
'
_output_shapes
:€€€€€€€€€
$
_user_specified_name
inputs/uid:U	Q
'
_output_shapes
:€€€€€€€€€
&
_user_specified_nameinputs/utag1:U
Q
'
_output_shapes
:€€€€€€€€€
&
_user_specified_nameinputs/utag2:UQ
'
_output_shapes
:€€€€€€€€€
&
_user_specified_nameinputs/utag3:UQ
'
_output_shapes
:€€€€€€€€€
&
_user_specified_nameinputs/utag4
і
Ґ
'__inference_model_layer_call_fn_2189133

inputs_iid	
inputs_itag1	
inputs_itag2	
inputs_itag3	
inputs_itag4	
inputs_itag4_cube
inputs_itag4_origin
inputs_itag4_square

inputs_uid	
inputs_utag1	
inputs_utag2	
inputs_utag3	
inputs_utag4	
unknown:
рУ	
	unknown_0:
рУ	
	unknown_1:
identityИҐStatefulPartitionedCall≠
StatefulPartitionedCallStatefulPartitionedCall
inputs_iidinputs_itag1inputs_itag2inputs_itag3inputs_itag4inputs_itag4_cubeinputs_itag4_origininputs_itag4_square
inputs_uidinputs_utag1inputs_utag2inputs_utag3inputs_utag4unknown	unknown_0	unknown_1*
Tin
2										*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_model_layer_call_and_return_conditional_losses_2188891o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Т
_input_shapesА
э:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
'
_output_shapes
:€€€€€€€€€
$
_user_specified_name
inputs/iid:UQ
'
_output_shapes
:€€€€€€€€€
&
_user_specified_nameinputs/itag1:UQ
'
_output_shapes
:€€€€€€€€€
&
_user_specified_nameinputs/itag2:UQ
'
_output_shapes
:€€€€€€€€€
&
_user_specified_nameinputs/itag3:UQ
'
_output_shapes
:€€€€€€€€€
&
_user_specified_nameinputs/itag4:ZV
'
_output_shapes
:€€€€€€€€€
+
_user_specified_nameinputs/itag4_cube:\X
'
_output_shapes
:€€€€€€€€€
-
_user_specified_nameinputs/itag4_origin:\X
'
_output_shapes
:€€€€€€€€€
-
_user_specified_nameinputs/itag4_square:SO
'
_output_shapes
:€€€€€€€€€
$
_user_specified_name
inputs/uid:U	Q
'
_output_shapes
:€€€€€€€€€
&
_user_specified_nameinputs/utag1:U
Q
'
_output_shapes
:€€€€€€€€€
&
_user_specified_nameinputs/utag2:UQ
'
_output_shapes
:€€€€€€€€€
&
_user_specified_nameinputs/utag3:UQ
'
_output_shapes
:€€€€€€€€€
&
_user_specified_nameinputs/utag4
“
Џ
#__inference__traced_restore_2189487
file_prefix4
&assignvariableop_fm_ranking_layer_bias:L
8assignvariableop_1_fm_ranking_layer_embedding_embeddings:
рУ	N
:assignvariableop_2_fm_ranking_layer_embedding_1_embeddings:
рУ	

identity_4ИҐAssignVariableOpҐAssignVariableOp_1ҐAssignVariableOp_2У
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*є
valueѓBђB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHx
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B B ≤
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*$
_output_shapes
::::*
dtypes
2[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:С
AssignVariableOpAssignVariableOp&assignvariableop_fm_ranking_layer_biasIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:І
AssignVariableOp_1AssignVariableOp8assignvariableop_1_fm_ranking_layer_embedding_embeddingsIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_2AssignVariableOp:assignvariableop_2_fm_ranking_layer_embedding_1_embeddingsIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 Ч

Identity_3Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^NoOp"/device:CPU:0*
T0*
_output_shapes
: U

Identity_4IdentityIdentity_3:output:0^NoOp_1*
T0*
_output_shapes
: Е
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2*"
_acd_function_control_output(*
_output_shapes
 "!

identity_4Identity_4:output:0*
_input_shapes

: : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_2:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
Ѓ
±
 __inference__traced_save_2189468
file_prefix4
0savev2_fm_ranking_layer_bias_read_readvariableopD
@savev2_fm_ranking_layer_embedding_embeddings_read_readvariableopF
Bsavev2_fm_ranking_layer_embedding_1_embeddings_read_readvariableop
savev2_const

identity_1ИҐMergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/partБ
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : У
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: Р
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*є
valueѓBђB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHu
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B B л
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:00savev2_fm_ranking_layer_bias_read_readvariableop@savev2_fm_ranking_layer_embedding_embeddings_read_readvariableopBsavev2_fm_ranking_layer_embedding_1_embeddings_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
2Р
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:Л
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*5
_input_shapes$
": ::
рУ	:
рУ	: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix: 

_output_shapes
::&"
 
_output_shapes
:
рУ	:&"
 
_output_shapes
:
рУ	:

_output_shapes
: 
£
«
'__inference_model_layer_call_fn_2189028
iid		
itag1		
itag2		
itag3		
itag4	

itag4_cube
itag4_origin
itag4_square
uid		
utag1		
utag2		
utag3		
utag4	
unknown:
рУ	
	unknown_0:
рУ	
	unknown_1:
identityИҐStatefulPartitionedCall“
StatefulPartitionedCallStatefulPartitionedCalliiditag1itag2itag3itag4
itag4_cubeitag4_originitag4_squareuidutag1utag2utag3utag4unknown	unknown_0	unknown_1*
Tin
2										*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_model_layer_call_and_return_conditional_losses_2188996o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Т
_input_shapesА
э:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€: : : 22
StatefulPartitionedCallStatefulPartitionedCall:L H
'
_output_shapes
:€€€€€€€€€

_user_specified_nameiid:NJ
'
_output_shapes
:€€€€€€€€€

_user_specified_nameitag1:NJ
'
_output_shapes
:€€€€€€€€€

_user_specified_nameitag2:NJ
'
_output_shapes
:€€€€€€€€€

_user_specified_nameitag3:NJ
'
_output_shapes
:€€€€€€€€€

_user_specified_nameitag4:SO
'
_output_shapes
:€€€€€€€€€
$
_user_specified_name
itag4_cube:UQ
'
_output_shapes
:€€€€€€€€€
&
_user_specified_nameitag4_origin:UQ
'
_output_shapes
:€€€€€€€€€
&
_user_specified_nameitag4_square:LH
'
_output_shapes
:€€€€€€€€€

_user_specified_nameuid:N	J
'
_output_shapes
:€€€€€€€€€

_user_specified_nameutag1:N
J
'
_output_shapes
:€€€€€€€€€

_user_specified_nameutag2:NJ
'
_output_shapes
:€€€€€€€€€

_user_specified_nameutag3:NJ
'
_output_shapes
:€€€€€€€€€

_user_specified_nameutag4
Б
≈
%__inference_signature_wrapper_2189297
iid		
itag1		
itag2		
itag3		
itag4	

itag4_cube
itag4_origin
itag4_square
uid		
utag1		
utag2		
utag3		
utag4	
unknown:
рУ	
	unknown_0:
рУ	
	unknown_1:
identityИҐStatefulPartitionedCall≤
StatefulPartitionedCallStatefulPartitionedCalliiditag1itag2itag3itag4
itag4_cubeitag4_originitag4_squareuidutag1utag2utag3utag4unknown	unknown_0	unknown_1*
Tin
2										*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *+
f&R$
"__inference__wrapped_model_2188781o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Т
_input_shapesА
э:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€: : : 22
StatefulPartitionedCallStatefulPartitionedCall:L H
'
_output_shapes
:€€€€€€€€€

_user_specified_nameiid:NJ
'
_output_shapes
:€€€€€€€€€

_user_specified_nameitag1:NJ
'
_output_shapes
:€€€€€€€€€

_user_specified_nameitag2:NJ
'
_output_shapes
:€€€€€€€€€

_user_specified_nameitag3:NJ
'
_output_shapes
:€€€€€€€€€

_user_specified_nameitag4:SO
'
_output_shapes
:€€€€€€€€€
$
_user_specified_name
itag4_cube:UQ
'
_output_shapes
:€€€€€€€€€
&
_user_specified_nameitag4_origin:UQ
'
_output_shapes
:€€€€€€€€€
&
_user_specified_nameitag4_square:LH
'
_output_shapes
:€€€€€€€€€

_user_specified_nameuid:N	J
'
_output_shapes
:€€€€€€€€€

_user_specified_nameutag1:N
J
'
_output_shapes
:€€€€€€€€€

_user_specified_nameutag2:NJ
'
_output_shapes
:€€€€€€€€€

_user_specified_nameutag3:NJ
'
_output_shapes
:€€€€€€€€€

_user_specified_nameutag4
÷@
У
M__inference_fm_ranking_layer_layer_call_and_return_conditional_losses_2188870

inputs	
inputs_1	
inputs_2	
inputs_3	
inputs_4	
inputs_5
inputs_6
inputs_7
inputs_8	
inputs_9	
	inputs_10	
	inputs_11	
	inputs_12	8
$embedding_1_embedding_lookup_2188829:
рУ	6
"embedding_embedding_lookup_2188834:
рУ	%
readvariableop_resource:
identityИҐReadVariableOpҐembedding/embedding_lookupҐembedding_1/embedding_lookupҐGfm_ranking_layer/embedding/embeddings/Regularizer/Square/ReadVariableOpҐIfm_ranking_layer/embedding_1/embeddings/Regularizer/Square/ReadVariableOpM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :»
concatConcatV2inputs_8inputsinputs_9	inputs_10	inputs_11	inputs_12inputs_1inputs_2inputs_3inputs_4concat/axis:output:0*
N
*
T0	*'
_output_shapes
:€€€€€€€€€
и
embedding_1/embedding_lookupResourceGather$embedding_1_embedding_lookup_2188829concat:output:0*
Tindices0	*7
_class-
+)loc:@embedding_1/embedding_lookup/2188829*+
_output_shapes
:€€€€€€€€€
*
dtype0«
%embedding_1/embedding_lookup/IdentityIdentity%embedding_1/embedding_lookup:output:0*
T0*7
_class-
+)loc:@embedding_1/embedding_lookup/2188829*+
_output_shapes
:€€€€€€€€€
Щ
'embedding_1/embedding_lookup/Identity_1Identity.embedding_1/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:€€€€€€€€€
в
embedding/embedding_lookupResourceGather"embedding_embedding_lookup_2188834concat:output:0*
Tindices0	*5
_class+
)'loc:@embedding/embedding_lookup/2188834*+
_output_shapes
:€€€€€€€€€
*
dtype0Ѕ
#embedding/embedding_lookup/IdentityIdentity#embedding/embedding_lookup:output:0*
T0*5
_class+
)'loc:@embedding/embedding_lookup/2188834*+
_output_shapes
:€€€€€€€€€
Х
%embedding/embedding_lookup/Identity_1Identity,embedding/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:€€€€€€€€€
W
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :О
SumSum0embedding_1/embedding_lookup/Identity_1:output:0Sum/reduction_indices:output:0*
T0*'
_output_shapes
:€€€€€€€€€v
SquareSquare.embedding/embedding_lookup/Identity_1:output:0*
T0*+
_output_shapes
:€€€€€€€€€
Y
Sum_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :l
Sum_1Sum
Square:y:0 Sum_1/reduction_indices:output:0*
T0*'
_output_shapes
:€€€€€€€€€Y
Sum_2/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :Р
Sum_2Sum.embedding/embedding_lookup/Identity_1:output:0 Sum_2/reduction_indices:output:0*
T0*'
_output_shapes
:€€€€€€€€€T
Square_1SquareSum_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€Z
SubSubSquare_1:y:0Sum_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€Y
Sum_3/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :z
Sum_3SumSub:z:0 Sum_3/reduction_indices:output:0*
T0*'
_output_shapes
:€€€€€€€€€*
	keep_dims(J
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?\
mulMulmul/x:output:0Sum_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0d
addAddV2ReadVariableOp:value:0Sum:output:0*
T0*'
_output_shapes
:€€€€€€€€€R
add_1AddV2add:z:0mul:z:0*
T0*'
_output_shapes
:€€€€€€€€€O
SigmoidSigmoid	add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ђ
Gfm_ranking_layer/embedding/embeddings/Regularizer/Square/ReadVariableOpReadVariableOp"embedding_embedding_lookup_2188834* 
_output_shapes
:
рУ	*
dtype0Њ
8fm_ranking_layer/embedding/embeddings/Regularizer/SquareSquareOfm_ranking_layer/embedding/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
рУ	И
7fm_ranking_layer/embedding/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       Ё
5fm_ranking_layer/embedding/embeddings/Regularizer/SumSum<fm_ranking_layer/embedding/embeddings/Regularizer/Square:y:0@fm_ranking_layer/embedding/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: |
7fm_ranking_layer/embedding/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<я
5fm_ranking_layer/embedding/embeddings/Regularizer/mulMul@fm_ranking_layer/embedding/embeddings/Regularizer/mul/x:output:0>fm_ranking_layer/embedding/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: ∞
Ifm_ranking_layer/embedding_1/embeddings/Regularizer/Square/ReadVariableOpReadVariableOp$embedding_1_embedding_lookup_2188829* 
_output_shapes
:
рУ	*
dtype0¬
:fm_ranking_layer/embedding_1/embeddings/Regularizer/SquareSquareQfm_ranking_layer/embedding_1/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
рУ	К
9fm_ranking_layer/embedding_1/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       г
7fm_ranking_layer/embedding_1/embeddings/Regularizer/SumSum>fm_ranking_layer/embedding_1/embeddings/Regularizer/Square:y:0Bfm_ranking_layer/embedding_1/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: ~
9fm_ranking_layer/embedding_1/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<е
7fm_ranking_layer/embedding_1/embeddings/Regularizer/mulMulBfm_ranking_layer/embedding_1/embeddings/Regularizer/mul/x:output:0@fm_ranking_layer/embedding_1/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€©
NoOpNoOp^ReadVariableOp^embedding/embedding_lookup^embedding_1/embedding_lookupH^fm_ranking_layer/embedding/embeddings/Regularizer/Square/ReadVariableOpJ^fm_ranking_layer/embedding_1/embeddings/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Т
_input_shapesА
э:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€: : : 2 
ReadVariableOpReadVariableOp28
embedding/embedding_lookupembedding/embedding_lookup2<
embedding_1/embedding_lookupembedding_1/embedding_lookup2Т
Gfm_ranking_layer/embedding/embeddings/Regularizer/Square/ReadVariableOpGfm_ranking_layer/embedding/embeddings/Regularizer/Square/ReadVariableOp2Ц
Ifm_ranking_layer/embedding_1/embeddings/Regularizer/Square/ReadVariableOpIfm_ranking_layer/embedding_1/embeddings/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:O	K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:O
K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
б*
Є
B__inference_model_layer_call_and_return_conditional_losses_2189098
iid		
itag1		
itag2		
itag3		
itag4	

itag4_cube
itag4_origin
itag4_square
uid		
utag1		
utag2		
utag3		
utag4	,
fm_ranking_layer_2189078:
рУ	,
fm_ranking_layer_2189080:
рУ	&
fm_ranking_layer_2189082:
identityИҐ(fm_ranking_layer/StatefulPartitionedCallҐGfm_ranking_layer/embedding/embeddings/Regularizer/Square/ReadVariableOpҐIfm_ranking_layer/embedding_1/embeddings/Regularizer/Square/ReadVariableOpЭ
(fm_ranking_layer/StatefulPartitionedCallStatefulPartitionedCalliiditag1itag2itag3itag4
itag4_cubeitag4_originitag4_squareuidutag1utag2utag3utag4fm_ranking_layer_2189078fm_ranking_layer_2189080fm_ranking_layer_2189082*
Tin
2										*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *V
fQRO
M__inference_fm_ranking_layer_layer_call_and_return_conditional_losses_2188870Ґ
Gfm_ranking_layer/embedding/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpfm_ranking_layer_2189080* 
_output_shapes
:
рУ	*
dtype0Њ
8fm_ranking_layer/embedding/embeddings/Regularizer/SquareSquareOfm_ranking_layer/embedding/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
рУ	И
7fm_ranking_layer/embedding/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       Ё
5fm_ranking_layer/embedding/embeddings/Regularizer/SumSum<fm_ranking_layer/embedding/embeddings/Regularizer/Square:y:0@fm_ranking_layer/embedding/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: |
7fm_ranking_layer/embedding/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<я
5fm_ranking_layer/embedding/embeddings/Regularizer/mulMul@fm_ranking_layer/embedding/embeddings/Regularizer/mul/x:output:0>fm_ranking_layer/embedding/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: §
Ifm_ranking_layer/embedding_1/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpfm_ranking_layer_2189078* 
_output_shapes
:
рУ	*
dtype0¬
:fm_ranking_layer/embedding_1/embeddings/Regularizer/SquareSquareQfm_ranking_layer/embedding_1/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
рУ	К
9fm_ranking_layer/embedding_1/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       г
7fm_ranking_layer/embedding_1/embeddings/Regularizer/SumSum>fm_ranking_layer/embedding_1/embeddings/Regularizer/Square:y:0Bfm_ranking_layer/embedding_1/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: ~
9fm_ranking_layer/embedding_1/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<е
7fm_ranking_layer/embedding_1/embeddings/Regularizer/mulMulBfm_ranking_layer/embedding_1/embeddings/Regularizer/mul/x:output:0@fm_ranking_layer/embedding_1/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: А
IdentityIdentity1fm_ranking_layer/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€З
NoOpNoOp)^fm_ranking_layer/StatefulPartitionedCallH^fm_ranking_layer/embedding/embeddings/Regularizer/Square/ReadVariableOpJ^fm_ranking_layer/embedding_1/embeddings/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Т
_input_shapesА
э:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€: : : 2T
(fm_ranking_layer/StatefulPartitionedCall(fm_ranking_layer/StatefulPartitionedCall2Т
Gfm_ranking_layer/embedding/embeddings/Regularizer/Square/ReadVariableOpGfm_ranking_layer/embedding/embeddings/Regularizer/Square/ReadVariableOp2Ц
Ifm_ranking_layer/embedding_1/embeddings/Regularizer/Square/ReadVariableOpIfm_ranking_layer/embedding_1/embeddings/Regularizer/Square/ReadVariableOp:L H
'
_output_shapes
:€€€€€€€€€

_user_specified_nameiid:NJ
'
_output_shapes
:€€€€€€€€€

_user_specified_nameitag1:NJ
'
_output_shapes
:€€€€€€€€€

_user_specified_nameitag2:NJ
'
_output_shapes
:€€€€€€€€€

_user_specified_nameitag3:NJ
'
_output_shapes
:€€€€€€€€€

_user_specified_nameitag4:SO
'
_output_shapes
:€€€€€€€€€
$
_user_specified_name
itag4_cube:UQ
'
_output_shapes
:€€€€€€€€€
&
_user_specified_nameitag4_origin:UQ
'
_output_shapes
:€€€€€€€€€
&
_user_specified_nameitag4_square:LH
'
_output_shapes
:€€€€€€€€€

_user_specified_nameuid:N	J
'
_output_shapes
:€€€€€€€€€

_user_specified_nameutag1:N
J
'
_output_shapes
:€€€€€€€€€

_user_specified_nameutag2:NJ
'
_output_shapes
:€€€€€€€€€

_user_specified_nameutag3:NJ
'
_output_shapes
:€€€€€€€€€

_user_specified_nameutag4
б*
Є
B__inference_model_layer_call_and_return_conditional_losses_2189063
iid		
itag1		
itag2		
itag3		
itag4	

itag4_cube
itag4_origin
itag4_square
uid		
utag1		
utag2		
utag3		
utag4	,
fm_ranking_layer_2189043:
рУ	,
fm_ranking_layer_2189045:
рУ	&
fm_ranking_layer_2189047:
identityИҐ(fm_ranking_layer/StatefulPartitionedCallҐGfm_ranking_layer/embedding/embeddings/Regularizer/Square/ReadVariableOpҐIfm_ranking_layer/embedding_1/embeddings/Regularizer/Square/ReadVariableOpЭ
(fm_ranking_layer/StatefulPartitionedCallStatefulPartitionedCalliiditag1itag2itag3itag4
itag4_cubeitag4_originitag4_squareuidutag1utag2utag3utag4fm_ranking_layer_2189043fm_ranking_layer_2189045fm_ranking_layer_2189047*
Tin
2										*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *V
fQRO
M__inference_fm_ranking_layer_layer_call_and_return_conditional_losses_2188870Ґ
Gfm_ranking_layer/embedding/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpfm_ranking_layer_2189045* 
_output_shapes
:
рУ	*
dtype0Њ
8fm_ranking_layer/embedding/embeddings/Regularizer/SquareSquareOfm_ranking_layer/embedding/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
рУ	И
7fm_ranking_layer/embedding/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       Ё
5fm_ranking_layer/embedding/embeddings/Regularizer/SumSum<fm_ranking_layer/embedding/embeddings/Regularizer/Square:y:0@fm_ranking_layer/embedding/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: |
7fm_ranking_layer/embedding/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<я
5fm_ranking_layer/embedding/embeddings/Regularizer/mulMul@fm_ranking_layer/embedding/embeddings/Regularizer/mul/x:output:0>fm_ranking_layer/embedding/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: §
Ifm_ranking_layer/embedding_1/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpfm_ranking_layer_2189043* 
_output_shapes
:
рУ	*
dtype0¬
:fm_ranking_layer/embedding_1/embeddings/Regularizer/SquareSquareQfm_ranking_layer/embedding_1/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
рУ	К
9fm_ranking_layer/embedding_1/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       г
7fm_ranking_layer/embedding_1/embeddings/Regularizer/SumSum>fm_ranking_layer/embedding_1/embeddings/Regularizer/Square:y:0Bfm_ranking_layer/embedding_1/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: ~
9fm_ranking_layer/embedding_1/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<е
7fm_ranking_layer/embedding_1/embeddings/Regularizer/mulMulBfm_ranking_layer/embedding_1/embeddings/Regularizer/mul/x:output:0@fm_ranking_layer/embedding_1/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: А
IdentityIdentity1fm_ranking_layer/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€З
NoOpNoOp)^fm_ranking_layer/StatefulPartitionedCallH^fm_ranking_layer/embedding/embeddings/Regularizer/Square/ReadVariableOpJ^fm_ranking_layer/embedding_1/embeddings/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Т
_input_shapesА
э:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€: : : 2T
(fm_ranking_layer/StatefulPartitionedCall(fm_ranking_layer/StatefulPartitionedCall2Т
Gfm_ranking_layer/embedding/embeddings/Regularizer/Square/ReadVariableOpGfm_ranking_layer/embedding/embeddings/Regularizer/Square/ReadVariableOp2Ц
Ifm_ranking_layer/embedding_1/embeddings/Regularizer/Square/ReadVariableOpIfm_ranking_layer/embedding_1/embeddings/Regularizer/Square/ReadVariableOp:L H
'
_output_shapes
:€€€€€€€€€

_user_specified_nameiid:NJ
'
_output_shapes
:€€€€€€€€€

_user_specified_nameitag1:NJ
'
_output_shapes
:€€€€€€€€€

_user_specified_nameitag2:NJ
'
_output_shapes
:€€€€€€€€€

_user_specified_nameitag3:NJ
'
_output_shapes
:€€€€€€€€€

_user_specified_nameitag4:SO
'
_output_shapes
:€€€€€€€€€
$
_user_specified_name
itag4_cube:UQ
'
_output_shapes
:€€€€€€€€€
&
_user_specified_nameitag4_origin:UQ
'
_output_shapes
:€€€€€€€€€
&
_user_specified_nameitag4_square:LH
'
_output_shapes
:€€€€€€€€€

_user_specified_nameuid:N	J
'
_output_shapes
:€€€€€€€€€

_user_specified_nameutag1:N
J
'
_output_shapes
:€€€€€€€€€

_user_specified_nameutag2:NJ
'
_output_shapes
:€€€€€€€€€

_user_specified_nameutag3:NJ
'
_output_shapes
:€€€€€€€€€

_user_specified_nameutag4
Є7
√
"__inference__wrapped_model_2188781
iid		
itag1		
itag2		
itag3		
itag4	

itag4_cube
itag4_origin
itag4_square
uid		
utag1		
utag2		
utag3		
utag4	O
;model_fm_ranking_layer_embedding_1_embedding_lookup_2188752:
рУ	M
9model_fm_ranking_layer_embedding_embedding_lookup_2188757:
рУ	<
.model_fm_ranking_layer_readvariableop_resource:
identityИҐ%model/fm_ranking_layer/ReadVariableOpҐ1model/fm_ranking_layer/embedding/embedding_lookupҐ3model/fm_ranking_layer/embedding_1/embedding_lookupd
"model/fm_ranking_layer/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :”
model/fm_ranking_layer/concatConcatV2uidiidutag1utag2utag3utag4itag1itag2itag3itag4+model/fm_ranking_layer/concat/axis:output:0*
N
*
T0	*'
_output_shapes
:€€€€€€€€€
ƒ
3model/fm_ranking_layer/embedding_1/embedding_lookupResourceGather;model_fm_ranking_layer_embedding_1_embedding_lookup_2188752&model/fm_ranking_layer/concat:output:0*
Tindices0	*N
_classD
B@loc:@model/fm_ranking_layer/embedding_1/embedding_lookup/2188752*+
_output_shapes
:€€€€€€€€€
*
dtype0М
<model/fm_ranking_layer/embedding_1/embedding_lookup/IdentityIdentity<model/fm_ranking_layer/embedding_1/embedding_lookup:output:0*
T0*N
_classD
B@loc:@model/fm_ranking_layer/embedding_1/embedding_lookup/2188752*+
_output_shapes
:€€€€€€€€€
«
>model/fm_ranking_layer/embedding_1/embedding_lookup/Identity_1IdentityEmodel/fm_ranking_layer/embedding_1/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:€€€€€€€€€
Њ
1model/fm_ranking_layer/embedding/embedding_lookupResourceGather9model_fm_ranking_layer_embedding_embedding_lookup_2188757&model/fm_ranking_layer/concat:output:0*
Tindices0	*L
_classB
@>loc:@model/fm_ranking_layer/embedding/embedding_lookup/2188757*+
_output_shapes
:€€€€€€€€€
*
dtype0Ж
:model/fm_ranking_layer/embedding/embedding_lookup/IdentityIdentity:model/fm_ranking_layer/embedding/embedding_lookup:output:0*
T0*L
_classB
@>loc:@model/fm_ranking_layer/embedding/embedding_lookup/2188757*+
_output_shapes
:€€€€€€€€€
√
<model/fm_ranking_layer/embedding/embedding_lookup/Identity_1IdentityCmodel/fm_ranking_layer/embedding/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:€€€€€€€€€
n
,model/fm_ranking_layer/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :”
model/fm_ranking_layer/SumSumGmodel/fm_ranking_layer/embedding_1/embedding_lookup/Identity_1:output:05model/fm_ranking_layer/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:€€€€€€€€€§
model/fm_ranking_layer/SquareSquareEmodel/fm_ranking_layer/embedding/embedding_lookup/Identity_1:output:0*
T0*+
_output_shapes
:€€€€€€€€€
p
.model/fm_ranking_layer/Sum_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :±
model/fm_ranking_layer/Sum_1Sum!model/fm_ranking_layer/Square:y:07model/fm_ranking_layer/Sum_1/reduction_indices:output:0*
T0*'
_output_shapes
:€€€€€€€€€p
.model/fm_ranking_layer/Sum_2/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :’
model/fm_ranking_layer/Sum_2SumEmodel/fm_ranking_layer/embedding/embedding_lookup/Identity_1:output:07model/fm_ranking_layer/Sum_2/reduction_indices:output:0*
T0*'
_output_shapes
:€€€€€€€€€В
model/fm_ranking_layer/Square_1Square%model/fm_ranking_layer/Sum_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€Я
model/fm_ranking_layer/SubSub#model/fm_ranking_layer/Square_1:y:0%model/fm_ranking_layer/Sum_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€p
.model/fm_ranking_layer/Sum_3/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :њ
model/fm_ranking_layer/Sum_3Summodel/fm_ranking_layer/Sub:z:07model/fm_ranking_layer/Sum_3/reduction_indices:output:0*
T0*'
_output_shapes
:€€€€€€€€€*
	keep_dims(a
model/fm_ranking_layer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?°
model/fm_ranking_layer/mulMul%model/fm_ranking_layer/mul/x:output:0%model/fm_ranking_layer/Sum_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€Р
%model/fm_ranking_layer/ReadVariableOpReadVariableOp.model_fm_ranking_layer_readvariableop_resource*
_output_shapes
:*
dtype0©
model/fm_ranking_layer/addAddV2-model/fm_ranking_layer/ReadVariableOp:value:0#model/fm_ranking_layer/Sum:output:0*
T0*'
_output_shapes
:€€€€€€€€€Ч
model/fm_ranking_layer/add_1AddV2model/fm_ranking_layer/add:z:0model/fm_ranking_layer/mul:z:0*
T0*'
_output_shapes
:€€€€€€€€€}
model/fm_ranking_layer/SigmoidSigmoid model/fm_ranking_layer/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€q
IdentityIdentity"model/fm_ranking_layer/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€Ў
NoOpNoOp&^model/fm_ranking_layer/ReadVariableOp2^model/fm_ranking_layer/embedding/embedding_lookup4^model/fm_ranking_layer/embedding_1/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Т
_input_shapesА
э:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€: : : 2N
%model/fm_ranking_layer/ReadVariableOp%model/fm_ranking_layer/ReadVariableOp2f
1model/fm_ranking_layer/embedding/embedding_lookup1model/fm_ranking_layer/embedding/embedding_lookup2j
3model/fm_ranking_layer/embedding_1/embedding_lookup3model/fm_ranking_layer/embedding_1/embedding_lookup:L H
'
_output_shapes
:€€€€€€€€€

_user_specified_nameiid:NJ
'
_output_shapes
:€€€€€€€€€

_user_specified_nameitag1:NJ
'
_output_shapes
:€€€€€€€€€

_user_specified_nameitag2:NJ
'
_output_shapes
:€€€€€€€€€

_user_specified_nameitag3:NJ
'
_output_shapes
:€€€€€€€€€

_user_specified_nameitag4:SO
'
_output_shapes
:€€€€€€€€€
$
_user_specified_name
itag4_cube:UQ
'
_output_shapes
:€€€€€€€€€
&
_user_specified_nameitag4_origin:UQ
'
_output_shapes
:€€€€€€€€€
&
_user_specified_nameitag4_square:LH
'
_output_shapes
:€€€€€€€€€

_user_specified_nameuid:N	J
'
_output_shapes
:€€€€€€€€€

_user_specified_nameutag1:N
J
'
_output_shapes
:€€€€€€€€€

_user_specified_nameutag2:NJ
'
_output_shapes
:€€€€€€€€€

_user_specified_nameutag3:NJ
'
_output_shapes
:€€€€€€€€€

_user_specified_nameutag4"џL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*щ
serving_defaultе
3
iid,
serving_default_iid:0	€€€€€€€€€
7
itag1.
serving_default_itag1:0	€€€€€€€€€
7
itag2.
serving_default_itag2:0	€€€€€€€€€
7
itag3.
serving_default_itag3:0	€€€€€€€€€
7
itag4.
serving_default_itag4:0	€€€€€€€€€
A

itag4_cube3
serving_default_itag4_cube:0€€€€€€€€€
E
itag4_origin5
serving_default_itag4_origin:0€€€€€€€€€
E
itag4_square5
serving_default_itag4_square:0€€€€€€€€€
3
uid,
serving_default_uid:0	€€€€€€€€€
7
utag1.
serving_default_utag1:0	€€€€€€€€€
7
utag2.
serving_default_utag2:0	€€€€€€€€€
7
utag3.
serving_default_utag3:0	€€€€€€€€€
7
utag4.
serving_default_utag4:0	€€€€€€€€€D
fm_ranking_layer0
StatefulPartitionedCall:0€€€€€€€€€tensorflow/serving/predict:ХВ
®
layer-0
layer-1
layer-2
layer-3
layer-4
layer-5
layer-6
layer-7
	layer-8

layer-9
layer-10
layer-11
layer-12
layer_with_weights-0
layer-13
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_network
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
‘
feature_names
bias
	embed
w
	variables
trainable_variables
regularization_losses
	keras_api
__call__
* &call_and_return_all_conditional_losses"
_tf_keras_layer
5
0
!1
"2"
trackable_list_wrapper
5
0
!1
"2"
trackable_list_wrapper
 "
trackable_list_wrapper
 
#non_trainable_variables

$layers
%metrics
&layer_regularization_losses
'layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
к2з
'__inference_model_layer_call_fn_2188900
'__inference_model_layer_call_fn_2189133
'__inference_model_layer_call_fn_2189156
'__inference_model_layer_call_fn_2189028ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
÷2”
B__inference_model_layer_call_and_return_conditional_losses_2189214
B__inference_model_layer_call_and_return_conditional_losses_2189272
B__inference_model_layer_call_and_return_conditional_losses_2189063
B__inference_model_layer_call_and_return_conditional_losses_2189098ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
ЃBЂ
"__inference__wrapped_model_2188781iiditag1itag2itag3itag4
itag4_cubeitag4_originitag4_squareuidutag1utag2utag3utag4"Ш
С≤Н
FullArgSpec
argsЪ 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
,
(serving_default"
signature_map
 "
trackable_list_wrapper
#:!2fm_ranking_layer/bias
µ
!
embeddings
)	variables
*trainable_variables
+regularization_losses
,	keras_api
-__call__
*.&call_and_return_all_conditional_losses"
_tf_keras_layer
µ
"
embeddings
/	variables
0trainable_variables
1regularization_losses
2	keras_api
3__call__
*4&call_and_return_all_conditional_losses"
_tf_keras_layer
5
0
!1
"2"
trackable_list_wrapper
5
0
!1
"2"
trackable_list_wrapper
.
50
61"
trackable_list_wrapper
≠
7non_trainable_variables

8layers
9metrics
:layer_regularization_losses
;layer_metrics
	variables
trainable_variables
regularization_losses
__call__
* &call_and_return_all_conditional_losses
& "call_and_return_conditional_losses"
_generic_user_object
№2ў
2__inference_fm_ranking_layer_layer_call_fn_2189332Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ч2ф
M__inference_fm_ranking_layer_layer_call_and_return_conditional_losses_2189390Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
9:7
рУ	2%fm_ranking_layer/embedding/embeddings
;:9
рУ	2'fm_ranking_layer/embedding_1/embeddings
 "
trackable_list_wrapper
Ж
0
1
2
3
4
5
6
7
	8

9
10
11
12
13"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ЂB®
%__inference_signature_wrapper_2189297iiditag1itag2itag3itag4
itag4_cubeitag4_originitag4_squareuidutag1utag2utag3utag4"Ф
Н≤Й
FullArgSpec
argsЪ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
'
!0"
trackable_list_wrapper
'
!0"
trackable_list_wrapper
'
50"
trackable_list_wrapper
≠
<non_trainable_variables

=layers
>metrics
?layer_regularization_losses
@layer_metrics
)	variables
*trainable_variables
+regularization_losses
-__call__
*.&call_and_return_all_conditional_losses
&."call_and_return_conditional_losses"
_generic_user_object
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
'
"0"
trackable_list_wrapper
'
"0"
trackable_list_wrapper
'
60"
trackable_list_wrapper
≠
Anon_trainable_variables

Blayers
Cmetrics
Dlayer_regularization_losses
Elayer_metrics
/	variables
0trainable_variables
1regularization_losses
3__call__
*4&call_and_return_all_conditional_losses
&4"call_and_return_conditional_losses"
_generic_user_object
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
і2±
__inference_loss_fn_0_2189413П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ 
і2±
__inference_loss_fn_1_2189424П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ 
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
50"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
60"
trackable_list_wrapper
 "
trackable_dict_wrapperЋ
"__inference__wrapped_model_2188781§"!„Ґ”
ЋҐ«
ƒ™ј
$
iidК
iid€€€€€€€€€	
(
itag1К
itag1€€€€€€€€€	
(
itag2К
itag2€€€€€€€€€	
(
itag3К
itag3€€€€€€€€€	
(
itag4К
itag4€€€€€€€€€	
2

itag4_cube$К!

itag4_cube€€€€€€€€€
6
itag4_origin&К#
itag4_origin€€€€€€€€€
6
itag4_square&К#
itag4_square€€€€€€€€€
$
uidК
uid€€€€€€€€€	
(
utag1К
utag1€€€€€€€€€	
(
utag2К
utag2€€€€€€€€€	
(
utag3К
utag3€€€€€€€€€	
(
utag4К
utag4€€€€€€€€€	
™ "C™@
>
fm_ranking_layer*К'
fm_ranking_layer€€€€€€€€€…
M__inference_fm_ranking_layer_layer_call_and_return_conditional_losses_2189390ч"!≤ҐЃ
¶ҐҐ
Я™Ы
+
iid$К!

inputs/iid€€€€€€€€€	
/
itag1&К#
inputs/itag1€€€€€€€€€	
/
itag2&К#
inputs/itag2€€€€€€€€€	
/
itag3&К#
inputs/itag3€€€€€€€€€	
/
itag4&К#
inputs/itag4€€€€€€€€€	
9

itag4_cube+К(
inputs/itag4_cube€€€€€€€€€
=
itag4_origin-К*
inputs/itag4_origin€€€€€€€€€
=
itag4_square-К*
inputs/itag4_square€€€€€€€€€
+
uid$К!

inputs/uid€€€€€€€€€	
/
utag1&К#
inputs/utag1€€€€€€€€€	
/
utag2&К#
inputs/utag2€€€€€€€€€	
/
utag3&К#
inputs/utag3€€€€€€€€€	
/
utag4&К#
inputs/utag4€€€€€€€€€	
™ ";Ґ8
1™.
,
output"К
0/output€€€€€€€€€
Ъ Ґ
2__inference_fm_ranking_layer_layer_call_fn_2189332л"!≤ҐЃ
¶ҐҐ
Я™Ы
+
iid$К!

inputs/iid€€€€€€€€€	
/
itag1&К#
inputs/itag1€€€€€€€€€	
/
itag2&К#
inputs/itag2€€€€€€€€€	
/
itag3&К#
inputs/itag3€€€€€€€€€	
/
itag4&К#
inputs/itag4€€€€€€€€€	
9

itag4_cube+К(
inputs/itag4_cube€€€€€€€€€
=
itag4_origin-К*
inputs/itag4_origin€€€€€€€€€
=
itag4_square-К*
inputs/itag4_square€€€€€€€€€
+
uid$К!

inputs/uid€€€€€€€€€	
/
utag1&К#
inputs/utag1€€€€€€€€€	
/
utag2&К#
inputs/utag2€€€€€€€€€	
/
utag3&К#
inputs/utag3€€€€€€€€€	
/
utag4&К#
inputs/utag4€€€€€€€€€	
™ "/™,
*
output К
output€€€€€€€€€<
__inference_loss_fn_0_2189413!Ґ

Ґ 
™ "К <
__inference_loss_fn_1_2189424"Ґ

Ґ 
™ "К л
B__inference_model_layer_call_and_return_conditional_losses_2189063§"!яҐџ
”Ґѕ
ƒ™ј
$
iidК
iid€€€€€€€€€	
(
itag1К
itag1€€€€€€€€€	
(
itag2К
itag2€€€€€€€€€	
(
itag3К
itag3€€€€€€€€€	
(
itag4К
itag4€€€€€€€€€	
2

itag4_cube$К!

itag4_cube€€€€€€€€€
6
itag4_origin&К#
itag4_origin€€€€€€€€€
6
itag4_square&К#
itag4_square€€€€€€€€€
$
uidК
uid€€€€€€€€€	
(
utag1К
utag1€€€€€€€€€	
(
utag2К
utag2€€€€€€€€€	
(
utag3К
utag3€€€€€€€€€	
(
utag4К
utag4€€€€€€€€€	
p 

 
™ ";Ґ8
1™.
,
output"К
0/output€€€€€€€€€
Ъ л
B__inference_model_layer_call_and_return_conditional_losses_2189098§"!яҐџ
”Ґѕ
ƒ™ј
$
iidК
iid€€€€€€€€€	
(
itag1К
itag1€€€€€€€€€	
(
itag2К
itag2€€€€€€€€€	
(
itag3К
itag3€€€€€€€€€	
(
itag4К
itag4€€€€€€€€€	
2

itag4_cube$К!

itag4_cube€€€€€€€€€
6
itag4_origin&К#
itag4_origin€€€€€€€€€
6
itag4_square&К#
itag4_square€€€€€€€€€
$
uidК
uid€€€€€€€€€	
(
utag1К
utag1€€€€€€€€€	
(
utag2К
utag2€€€€€€€€€	
(
utag3К
utag3€€€€€€€€€	
(
utag4К
utag4€€€€€€€€€	
p

 
™ ";Ґ8
1™.
,
output"К
0/output€€€€€€€€€
Ъ ∆
B__inference_model_layer_call_and_return_conditional_losses_2189214€"!ЇҐґ
ЃҐ™
Я™Ы
+
iid$К!

inputs/iid€€€€€€€€€	
/
itag1&К#
inputs/itag1€€€€€€€€€	
/
itag2&К#
inputs/itag2€€€€€€€€€	
/
itag3&К#
inputs/itag3€€€€€€€€€	
/
itag4&К#
inputs/itag4€€€€€€€€€	
9

itag4_cube+К(
inputs/itag4_cube€€€€€€€€€
=
itag4_origin-К*
inputs/itag4_origin€€€€€€€€€
=
itag4_square-К*
inputs/itag4_square€€€€€€€€€
+
uid$К!

inputs/uid€€€€€€€€€	
/
utag1&К#
inputs/utag1€€€€€€€€€	
/
utag2&К#
inputs/utag2€€€€€€€€€	
/
utag3&К#
inputs/utag3€€€€€€€€€	
/
utag4&К#
inputs/utag4€€€€€€€€€	
p 

 
™ ";Ґ8
1™.
,
output"К
0/output€€€€€€€€€
Ъ ∆
B__inference_model_layer_call_and_return_conditional_losses_2189272€"!ЇҐґ
ЃҐ™
Я™Ы
+
iid$К!

inputs/iid€€€€€€€€€	
/
itag1&К#
inputs/itag1€€€€€€€€€	
/
itag2&К#
inputs/itag2€€€€€€€€€	
/
itag3&К#
inputs/itag3€€€€€€€€€	
/
itag4&К#
inputs/itag4€€€€€€€€€	
9

itag4_cube+К(
inputs/itag4_cube€€€€€€€€€
=
itag4_origin-К*
inputs/itag4_origin€€€€€€€€€
=
itag4_square-К*
inputs/itag4_square€€€€€€€€€
+
uid$К!

inputs/uid€€€€€€€€€	
/
utag1&К#
inputs/utag1€€€€€€€€€	
/
utag2&К#
inputs/utag2€€€€€€€€€	
/
utag3&К#
inputs/utag3€€€€€€€€€	
/
utag4&К#
inputs/utag4€€€€€€€€€	
p

 
™ ";Ґ8
1™.
,
output"К
0/output€€€€€€€€€
Ъ ƒ
'__inference_model_layer_call_fn_2188900Ш"!яҐџ
”Ґѕ
ƒ™ј
$
iidК
iid€€€€€€€€€	
(
itag1К
itag1€€€€€€€€€	
(
itag2К
itag2€€€€€€€€€	
(
itag3К
itag3€€€€€€€€€	
(
itag4К
itag4€€€€€€€€€	
2

itag4_cube$К!

itag4_cube€€€€€€€€€
6
itag4_origin&К#
itag4_origin€€€€€€€€€
6
itag4_square&К#
itag4_square€€€€€€€€€
$
uidК
uid€€€€€€€€€	
(
utag1К
utag1€€€€€€€€€	
(
utag2К
utag2€€€€€€€€€	
(
utag3К
utag3€€€€€€€€€	
(
utag4К
utag4€€€€€€€€€	
p 

 
™ "/™,
*
output К
output€€€€€€€€€ƒ
'__inference_model_layer_call_fn_2189028Ш"!яҐџ
”Ґѕ
ƒ™ј
$
iidК
iid€€€€€€€€€	
(
itag1К
itag1€€€€€€€€€	
(
itag2К
itag2€€€€€€€€€	
(
itag3К
itag3€€€€€€€€€	
(
itag4К
itag4€€€€€€€€€	
2

itag4_cube$К!

itag4_cube€€€€€€€€€
6
itag4_origin&К#
itag4_origin€€€€€€€€€
6
itag4_square&К#
itag4_square€€€€€€€€€
$
uidК
uid€€€€€€€€€	
(
utag1К
utag1€€€€€€€€€	
(
utag2К
utag2€€€€€€€€€	
(
utag3К
utag3€€€€€€€€€	
(
utag4К
utag4€€€€€€€€€	
p

 
™ "/™,
*
output К
output€€€€€€€€€Я
'__inference_model_layer_call_fn_2189133у"!ЇҐґ
ЃҐ™
Я™Ы
+
iid$К!

inputs/iid€€€€€€€€€	
/
itag1&К#
inputs/itag1€€€€€€€€€	
/
itag2&К#
inputs/itag2€€€€€€€€€	
/
itag3&К#
inputs/itag3€€€€€€€€€	
/
itag4&К#
inputs/itag4€€€€€€€€€	
9

itag4_cube+К(
inputs/itag4_cube€€€€€€€€€
=
itag4_origin-К*
inputs/itag4_origin€€€€€€€€€
=
itag4_square-К*
inputs/itag4_square€€€€€€€€€
+
uid$К!

inputs/uid€€€€€€€€€	
/
utag1&К#
inputs/utag1€€€€€€€€€	
/
utag2&К#
inputs/utag2€€€€€€€€€	
/
utag3&К#
inputs/utag3€€€€€€€€€	
/
utag4&К#
inputs/utag4€€€€€€€€€	
p 

 
™ "/™,
*
output К
output€€€€€€€€€Я
'__inference_model_layer_call_fn_2189156у"!ЇҐґ
ЃҐ™
Я™Ы
+
iid$К!

inputs/iid€€€€€€€€€	
/
itag1&К#
inputs/itag1€€€€€€€€€	
/
itag2&К#
inputs/itag2€€€€€€€€€	
/
itag3&К#
inputs/itag3€€€€€€€€€	
/
itag4&К#
inputs/itag4€€€€€€€€€	
9

itag4_cube+К(
inputs/itag4_cube€€€€€€€€€
=
itag4_origin-К*
inputs/itag4_origin€€€€€€€€€
=
itag4_square-К*
inputs/itag4_square€€€€€€€€€
+
uid$К!

inputs/uid€€€€€€€€€	
/
utag1&К#
inputs/utag1€€€€€€€€€	
/
utag2&К#
inputs/utag2€€€€€€€€€	
/
utag3&К#
inputs/utag3€€€€€€€€€	
/
utag4&К#
inputs/utag4€€€€€€€€€	
p

 
™ "/™,
*
output К
output€€€€€€€€€«
%__inference_signature_wrapper_2189297Э"!–Ґћ
Ґ 
ƒ™ј
$
iidК
iid€€€€€€€€€	
(
itag1К
itag1€€€€€€€€€	
(
itag2К
itag2€€€€€€€€€	
(
itag3К
itag3€€€€€€€€€	
(
itag4К
itag4€€€€€€€€€	
2

itag4_cube$К!

itag4_cube€€€€€€€€€
6
itag4_origin&К#
itag4_origin€€€€€€€€€
6
itag4_square&К#
itag4_square€€€€€€€€€
$
uidК
uid€€€€€€€€€	
(
utag1К
utag1€€€€€€€€€	
(
utag2К
utag2€€€€€€€€€	
(
utag3К
utag3€€€€€€€€€	
(
utag4К
utag4€€€€€€€€€	"C™@
>
fm_ranking_layer*К'
fm_ranking_layer€€€€€€€€€