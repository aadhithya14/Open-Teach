/************************************************************************************
Copyright : Copyright (c) Facebook Technologies, LLC and its affiliates. All rights reserved.

Your use of this SDK or tool is subject to the Oculus SDK License Agreement, available at
https://developer.oculus.com/licenses/oculussdk/

Unless required by applicable law or agreed to in writing, the Utilities SDK distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
ANY KIND, either express or implied. See the License for the specific language governing
permissions and limitations under the License.
************************************************************************************/

using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// Primitive type serialization
/// </summary>
namespace Oculus.Interaction.Input
{
    public static class Constants
    {
        public const int NUM_HAND_JOINTS = (int)HandJointId.HandEnd;
        public const int NUM_FINGERS = 5;
    }

    public enum Handedness
    {
        Left = 0,
        Right = 1,
    }

    public enum HandFinger
    {
        Invalid = -1,
        Thumb = 0,
        Index = 1,
        Middle = 2,
        Ring = 3,
        Pinky = 4,
        Max = 4
    }

    [Flags]
    public enum HandFingerFlags
    {
        None = 0,
        Thumb = 1 << 0,
        Index = 1 << 1,
        Middle = 1 << 2,
        Ring = 1 << 3,
        Pinky = 1 << 4,
        All = (1 << 5) - 1
    }

    [Flags]
    public enum HandFingerJointFlags
    {
        None = 0,
        Thumb0 = 1 << HandJointId.HandThumb0,
        Thumb1 = 1 << HandJointId.HandThumb1,
        Thumb2 = 1 << HandJointId.HandThumb2,
        Thumb3 = 1 << HandJointId.HandThumb3,
        Index1 = 1 << HandJointId.HandIndex1,
        Index2 = 1 << HandJointId.HandIndex2,
        Index3 = 1 << HandJointId.HandIndex3,
        Middle1 = 1 << HandJointId.HandMiddle1,
        Middle2 = 1 << HandJointId.HandMiddle2,
        Middle3 = 1 << HandJointId.HandMiddle3,
        Ring1 = 1 << HandJointId.HandRing1,
        Ring2 = 1 << HandJointId.HandRing2,
        Ring3 = 1 << HandJointId.HandRing3,
        Pinky0 = 1 << HandJointId.HandPinky0,
        Pinky1 = 1 << HandJointId.HandPinky1,
        Pinky2 = 1 << HandJointId.HandPinky2,
        Pinky3 = 1 << HandJointId.HandPinky3,
    }

    public static class HandFingerUtils
    {
        public static HandFingerFlags ToFlags(HandFinger handFinger)
        {
            return (HandFingerFlags)(1 << (int)handFinger);
        }
    }

    public enum HandJointId
    {
        Invalid = -1,

        // hand bones
        HandStart = 0,
        HandWristRoot = HandStart + 0, // root frame of the hand, where the wrist is located
        HandForearmStub = HandStart + 1, // frame for user's forearm
        HandThumb0 = HandStart + 2, // thumb trapezium bone
        HandThumb1 = HandStart + 3, // thumb metacarpal bone
        HandThumb2 = HandStart + 4, // thumb proximal phalange bone
        HandThumb3 = HandStart + 5, // thumb distal phalange bone
        HandIndex1 = HandStart + 6, // index proximal phalange bone
        HandIndex2 = HandStart + 7, // index intermediate phalange bone
        HandIndex3 = HandStart + 8, // index distal phalange bone
        HandMiddle1 = HandStart + 9, // middle proximal phalange bone
        HandMiddle2 = HandStart + 10, // middle intermediate phalange bone
        HandMiddle3 = HandStart + 11, // middle distal phalange bone
        HandRing1 = HandStart + 12, // ring proximal phalange bone
        HandRing2 = HandStart + 13, // ring intermediate phalange bone
        HandRing3 = HandStart + 14, // ring distal phalange bone
        HandPinky0 = HandStart + 15, // pinky metacarpal bone
        HandPinky1 = HandStart + 16, // pinky proximal phalange bone
        HandPinky2 = HandStart + 17, // pinky intermediate phalange bone
        HandPinky3 = HandStart + 18, // pinky distal phalange bone
        HandMaxSkinnable = HandStart + 19,
        // Bone tips are position only.
        // They are not used for skinning but are useful for hit-testing.
        // NOTE: HandThumbTip == HandMaxSkinnable since the extended tips need to be contiguous
        HandThumbTip = HandMaxSkinnable + 0, // tip of the thumb
        HandIndexTip = HandMaxSkinnable + 1, // tip of the index finger
        HandMiddleTip = HandMaxSkinnable + 2, // tip of the middle finger
        HandRingTip = HandMaxSkinnable + 3, // tip of the ring finger
        HandPinkyTip = HandMaxSkinnable + 4, // tip of the pinky
        HandEnd = HandMaxSkinnable + 5,
    }

    public class HandJointUtils
    {
        public static List<HandJointId[]> FingerToJointList = new List<HandJointId[]>()
        {
            new[] {HandJointId.HandThumb0,HandJointId.HandThumb1,HandJointId.HandThumb2,HandJointId.HandThumb3},
            new[] {HandJointId.HandIndex1, HandJointId.HandIndex2, HandJointId.HandIndex3},
            new[] {HandJointId.HandMiddle1, HandJointId.HandMiddle2, HandJointId.HandMiddle3},
            new[] {HandJointId.HandRing1,HandJointId.HandRing2,HandJointId.HandRing3},
            new[] {HandJointId.HandPinky0, HandJointId.HandPinky1, HandJointId.HandPinky2, HandJointId.HandPinky3}
        };

        public static HandJointId[] JointParentList = new[]
        {
            HandJointId.Invalid,
            HandJointId.HandStart,
            HandJointId.HandStart,
            HandJointId.HandThumb0,
            HandJointId.HandThumb1,
            HandJointId.HandThumb2,
            HandJointId.HandStart,
            HandJointId.HandIndex1,
            HandJointId.HandIndex2,
            HandJointId.HandStart,
            HandJointId.HandMiddle1,
            HandJointId.HandMiddle2,
            HandJointId.HandStart,
            HandJointId.HandRing1,
            HandJointId.HandRing2,
            HandJointId.HandStart,
            HandJointId.HandPinky0,
            HandJointId.HandPinky1,
            HandJointId.HandPinky2,
            HandJointId.HandThumb3,
            HandJointId.HandIndex3,
            HandJointId.HandMiddle3,
            HandJointId.HandRing3,
            HandJointId.HandPinky3
        };

        public static HandJointId[][] JointChildrenList = new[]
        {
            new []
            {
                HandJointId.HandThumb0,
                HandJointId.HandIndex1,
                HandJointId.HandMiddle1,
                HandJointId.HandRing1,
                HandJointId.HandPinky0
            },
            new HandJointId[0],
            new []{ HandJointId.HandThumb1 },
            new []{ HandJointId.HandThumb2 },
            new []{ HandJointId.HandThumb3 },
            new []{ HandJointId.HandThumbTip },
            new []{ HandJointId.HandIndex2 },
            new []{ HandJointId.HandIndex3 },
            new []{ HandJointId.HandIndexTip },
            new []{ HandJointId.HandMiddle2 },
            new []{ HandJointId.HandMiddle3 },
            new []{ HandJointId.HandMiddleTip },
            new []{ HandJointId.HandRing2 },
            new []{ HandJointId.HandRing3 },
            new []{ HandJointId.HandRingTip },
            new []{ HandJointId.HandPinky1 },
            new []{ HandJointId.HandPinky2 },
            new []{ HandJointId.HandPinky3 },
            new []{ HandJointId.HandPinkyTip },
            new HandJointId[0],
            new HandJointId[0],
            new HandJointId[0],
            new HandJointId[0],
            new HandJointId[0]
        };

        public static List<HandJointId> JointIds = new List<HandJointId>()
        {
            HandJointId.HandIndex1,
            HandJointId.HandIndex2,
            HandJointId.HandIndex3,
            HandJointId.HandMiddle1,
            HandJointId.HandMiddle2,
            HandJointId.HandMiddle3,
            HandJointId.HandRing1,
            HandJointId.HandRing2,
            HandJointId.HandRing3,
            HandJointId.HandPinky0,
            HandJointId.HandPinky1,
            HandJointId.HandPinky2,
            HandJointId.HandPinky3,
            HandJointId.HandThumb0,
            HandJointId.HandThumb1,
            HandJointId.HandThumb2,
            HandJointId.HandThumb3
        };

        private static readonly HandJointId[] _handFingerProximals =
        {
            HandJointId.HandThumb2, HandJointId.HandIndex1, HandJointId.HandMiddle1,
            HandJointId.HandRing1, HandJointId.HandPinky1
        };

        public static HandJointId GetHandFingerTip(HandFinger finger)
        {
            return HandJointId.HandMaxSkinnable + (int)finger;
        }

        /// <summary>
        /// Returns the "proximal" JointId for the given finger.
        /// This is commonly known as the Knuckle.
        /// For fingers, proximal is the join with index 1; eg HandIndex1.
        /// For thumb, proximal is the joint with index 2; eg HandThumb2.
        /// </summary>
        public static HandJointId GetHandFingerProximal(HandFinger finger)
        {
            return _handFingerProximals[(int)finger];
        }
    }

    public struct HandSkeletonJoint
    {
        /// <summary>
        /// Id of the parent joint in the skeleton hierarchy. Must always have a lower index than
        /// this joint.
        /// </summary>
        public int parent;

        /// <summary>
        /// Stores the pose of the joint, in local space.
        /// </summary>
        public Pose pose;
    }

    public interface IReadOnlyHandSkeletonJointList
    {
        ref readonly HandSkeletonJoint this[int jointId] { get; }
    }

    public interface IReadOnlyHandSkeleton
    {
        IReadOnlyHandSkeletonJointList Joints { get; }
    }

    public interface ICopyFrom<in TSelfType>
    {
        void CopyFrom(TSelfType source);
    }

    public class ReadOnlyHandJointPoses : IReadOnlyList<Pose>
    {
        private Pose[] _poses;

        public ReadOnlyHandJointPoses(Pose[] poses)
        {
            _poses = poses;
        }

        public IEnumerator<Pose> GetEnumerator()
        {
            foreach (var pose in _poses)
            {
                yield return pose;
            }
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }

        public static ReadOnlyHandJointPoses Empty { get; } = new ReadOnlyHandJointPoses(Array.Empty<Pose>());

        public int Count => _poses.Length;

        public Pose this[int index] => _poses[index];

        public ref readonly Pose this[HandJointId index] => ref _poses[(int)index];
    }

    public class HandSkeleton : IReadOnlyHandSkeleton, IReadOnlyHandSkeletonJointList
    {
        public HandSkeletonJoint[] joints = new HandSkeletonJoint[Constants.NUM_HAND_JOINTS];
        public IReadOnlyHandSkeletonJointList Joints => this;
        public ref readonly HandSkeletonJoint this[int jointId] => ref joints[jointId];
    }
}
