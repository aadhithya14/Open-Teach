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
using UnityEngine;

namespace Oculus.Interaction
{
    public enum PointerEvent
    {
        Hover,
        Unhover,
        Select,
        Unselect,
        Move,
        Cancel
    }

    public struct PointerArgs
    {
        public int Identifier { get; }
        public PointerEvent PointerEvent { get; }
        public Pose Pose { get; }

        public PointerArgs(int identifier, PointerEvent pointerEvent, Pose pose)
        {
            this.Identifier = identifier;
            this.PointerEvent = pointerEvent;
            this.Pose = pose;
        }
    }

    public interface IPointable
    {
        event Action<PointerArgs> WhenPointerEventRaised;
    }

    public interface IPointableElement : IPointable
    {
        void ProcessPointerEvent(PointerArgs args);
    }
}
